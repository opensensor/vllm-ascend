# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host tests for GLM's contiguous resident packed-expert bank."""

from types import SimpleNamespace

import torch

from tools.deepseek_w2.w2_format import unpack_codes
from vllm_ascend.models.glm5next_w2.model import (
    GLM_NZ_INPUT_TILE,
    _pack_codes_nz,
    _PackedW2ExpertBank,
    _release_grouped_compaction_cache,
)


def _fill_local_bank(bank: _PackedW2ExpertBank) -> None:
    for expert_id in range(bank.local_expert_offset, bank.local_expert_offset + bank.num_local_experts):
        expert = bank[expert_id]
        value = expert_id + 1
        expert.gate_packed = torch.full((32, 32), value, dtype=torch.uint8)
        expert.gate_scale = torch.full((1, 2), float(value))
        expert.up_packed = torch.full((32, 32), value + 1, dtype=torch.uint8)
        expert.up_scale = torch.full((1, 2), float(value + 1))
        expert.down_packed = torch.full((64, 16), value + 2, dtype=torch.uint8)
        expert.down_scale = torch.full((2, 1), float(value + 2))


def test_resident_bank_compacts_local_experts_and_preserves_views():
    bank = _PackedW2ExpertBank(
        hidden=64,
        inter=32,
        num_experts=4,
        local_expert_offset=1,
        num_local_experts=2,
        offload_to_cpu=False,
    )
    _fill_local_bank(bank)

    bank.finalize_grouped_storage()

    assert bank.grouped_ready
    assert bank.gate_packed_bank.shape == (2, 32, 32)
    assert bank.down_scale_bank.shape == (2, 2, 1)
    assert bank[1].gate_packed.untyped_storage().data_ptr() == bank.gate_packed_bank.untyped_storage().data_ptr()
    assert bank[2].down_scale.untyped_storage().data_ptr() == bank.down_scale_bank.untyped_storage().data_ptr()
    assert torch.equal(bank[2].gate_packed, bank.gate_packed_bank[1])


def test_resident_bank_streams_directly_into_final_grouped_storage():
    bank = _PackedW2ExpertBank(
        hidden=64,
        inter=32,
        num_experts=4,
        local_expert_offset=1,
        num_local_experts=2,
        offload_to_cpu=False,
    )
    first = torch.full((32, 32), 7, dtype=torch.uint8)
    second = torch.full((32, 32), 9, dtype=torch.uint8)

    bank.place_resident_tensor(1, "gate_packed", first, device="cpu")
    grouped_storage = bank.gate_packed_bank.untyped_storage().data_ptr()
    bank.place_resident_tensor(2, "gate_packed", second, device="cpu")

    assert bank.gate_packed_bank.shape == (2, 32, 32)
    assert bank.gate_packed_bank.untyped_storage().data_ptr() == grouped_storage
    assert bank[1].gate_packed.untyped_storage().data_ptr() == grouped_storage
    assert bank[2].gate_packed.untyped_storage().data_ptr() == grouped_storage
    assert torch.equal(bank.gate_packed_bank[0], first)
    assert torch.equal(bank.gate_packed_bank[1], second)

    replacement = torch.full((32, 32), 11, dtype=torch.uint8)
    bank.place_resident_tensor(1, "gate_packed", replacement, device="cpu")

    assert bank.gate_packed_bank.untyped_storage().data_ptr() == grouped_storage
    assert torch.equal(bank.gate_packed_bank[0], replacement)


def test_resident_gate_up_share_one_allocation_and_rebuild_together():
    bank = _PackedW2ExpertBank(
        hidden=256,
        inter=128,
        num_experts=2,
        local_expert_offset=0,
        num_local_experts=2,
        offload_to_cpu=False,
    )
    for expert_id in range(2):
        for projection, value in (("gate", 3), ("up", 5)):
            bank.place_resident_tensor(
                expert_id,
                f"{projection}_packed",
                torch.full((128, 64), value + expert_id, dtype=torch.uint8),
                device="cpu",
            )
            bank.place_resident_tensor(
                expert_id,
                f"{projection}_scale",
                torch.full((4, 8), value + expert_id),
                device="cpu",
            )

    for kind in ("packed", "scale"):
        fused = getattr(bank, f"gate_up_{kind}_bank")
        gate = getattr(bank, f"gate_{kind}_bank")
        up = getattr(bank, f"up_{kind}_bank")
        assert fused.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()
        assert fused.untyped_storage().data_ptr() == up.untyped_storage().data_ptr()
        assert torch.equal(fused[:, : gate.shape[1]], gate)
        assert torch.equal(fused[:, gate.shape[1] :], up)

    bank.place_resident_tensor(0, "gate_packed", torch.full((128, 128), 7, dtype=torch.uint8), device="cpu")
    assert bank[0].up_packed is None
    assert bank[1].gate_packed is None
    assert bank[1].up_packed is None
    assert bank.gate_up_packed_bank.shape == (2, 256, 128)
    bank.place_resident_tensor(0, "up_packed", torch.full((128, 128), 9, dtype=torch.uint8), device="cpu")
    assert torch.equal(bank.gate_up_packed_bank[0, :128], bank[0].gate_packed)
    assert torch.equal(bank.gate_up_packed_bank[0, 128:], bank[0].up_packed)


def test_resident_bank_rebuilds_projection_for_overlay_width_change():
    bank = _PackedW2ExpertBank(
        hidden=64,
        inter=32,
        num_experts=2,
        local_expert_offset=0,
        num_local_experts=2,
        offload_to_cpu=False,
        layer_key="layers.7",
    )
    w2_first = torch.full((64, 8), 2, dtype=torch.uint8)
    w2_second = torch.full((64, 8), 3, dtype=torch.uint8)
    bank.place_resident_tensor(0, "down_packed", w2_first, device="cpu")
    bank.place_resident_tensor(1, "down_packed", w2_second, device="cpu")

    w4_first = torch.full((64, 16), 4, dtype=torch.uint8)
    bank.place_resident_tensor(0, "down_packed", w4_first, device="cpu")

    assert bank.down_packed_bank.shape == (2, 64, 16)
    assert torch.equal(bank[0].down_packed, w4_first)
    assert bank[1].down_packed is None

    w4_second = torch.full((64, 16), 5, dtype=torch.uint8)
    bank.place_resident_tensor(1, "down_packed", w4_second, device="cpu")

    assert torch.equal(bank.down_packed_bank[1], w4_second)


def test_nz_packed_codes_preserve_all_w2_w4_values():
    for bits in (2, 4):
        torch.manual_seed(bits)
        n, k = 32, 2 * GLM_NZ_INPUT_TILE
        codes_per_byte = 8 // bits
        codes = torch.randint(0, 256, (n, k // codes_per_byte), dtype=torch.uint8)
        packed = _pack_codes_nz(codes, k)
        assert packed.shape == codes.shape
        assert packed.dtype == torch.uint8

        tile_bytes = 16 * GLM_NZ_INPUT_TILE // codes_per_byte
        tiles = packed.view(n // 16, k // GLM_NZ_INPUT_TILE, tile_bytes)
        unsigned = unpack_codes(codes, k, bits).to(torch.int16) & ((1 << bits) - 1)
        for n_tile in range(n // 16):
            for k_tile in range(k // GLM_NZ_INPUT_TILE):
                tile = tiles[n_tile, k_tile]
                decoded = torch.stack(
                    [(tile >> (bits * field)) & ((1 << bits) - 1) for field in range(codes_per_byte)]
                ).flatten()
                expected = unsigned[
                    n_tile * 16 : (n_tile + 1) * 16,
                    k_tile * GLM_NZ_INPUT_TILE : (k_tile + 1) * GLM_NZ_INPUT_TILE,
                ]
                assert torch.equal(decoded, expected.t().reshape(-1).to(torch.uint8))


def test_resident_bank_marks_nz_codes_with_int8_storage():
    bank = _PackedW2ExpertBank(
        hidden=256,
        inter=128,
        num_experts=1,
        local_expert_offset=0,
        num_local_experts=1,
        offload_to_cpu=False,
        nz_packed_codes=True,
    )
    codes = torch.randint(0, 256, (128, 128), dtype=torch.uint8)
    bank.place_resident_tensor(0, "gate_packed", codes, device="cpu")
    assert bank.gate_packed_bank.dtype == torch.int8
    assert torch.equal(bank.gate_packed_bank[0].view(torch.uint8), _pack_codes_nz(codes, 256))


def test_offloaded_bank_keeps_selected_expert_staging_layout():
    bank = _PackedW2ExpertBank(
        hidden=64,
        inter=32,
        num_experts=2,
        local_expert_offset=0,
        num_local_experts=2,
        offload_to_cpu=True,
    )
    _fill_local_bank(bank)

    bank.finalize_grouped_storage()

    assert not bank.grouped_ready
    assert not hasattr(bank, "gate_packed_bank")


def test_grouped_compaction_releases_npu_cache_once(monkeypatch):
    calls = 0

    def empty_cache() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch, "npu", SimpleNamespace(empty_cache=empty_cache), raising=False)
    npu_bank = SimpleNamespace(
        grouped_ready=True,
        gate_packed_bank=SimpleNamespace(device=SimpleNamespace(type="npu")),
    )
    cpu_bank = SimpleNamespace(
        grouped_ready=True,
        gate_packed_bank=SimpleNamespace(device=SimpleNamespace(type="cpu")),
    )

    _release_grouped_compaction_cache([npu_bank, cpu_bank, npu_bank])

    assert calls == 1


def test_grouped_compaction_skips_cache_release_without_npu_bank(monkeypatch):
    calls = 0

    def empty_cache() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch, "npu", SimpleNamespace(empty_cache=empty_cache), raising=False)
    cpu_bank = SimpleNamespace(
        grouped_ready=True,
        gate_packed_bank=SimpleNamespace(device=SimpleNamespace(type="cpu")),
    )

    _release_grouped_compaction_cache([cpu_bank])

    assert calls == 0
