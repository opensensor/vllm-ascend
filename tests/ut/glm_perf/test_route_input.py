# SPDX-License-Identifier: Apache-2.0
"""Routed-input capacity, launch ABI, and replay scratch ownership."""

import json

import pytest
import torch

from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE, route_input_shapes


def test_routed_input_requires_matching_consumer_layout_before_build(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="routed input packing requires paired"):
        build(output, tmp_path, tmp_path, route_packed_input=True)
    assert not output.exists()


@pytest.mark.parametrize("counts", [(0, 0, 0), (1, 31, 32), (30, 1, 0, 62), (63, 62, 33), (5120, 0, 0)])
def test_batch_slots_are_disjoint_and_bounded_even_with_empty_experts(counts):
    # Check slot allocation without allocating memory proportional to all experts.
    routes = max(17, sum(counts))
    geometry = FusedGeometry(routes, 1, len(counts), 4096, 256, 3, 3, 4)
    shape, scales = route_input_shapes(geometry)
    first, used = 0, set()
    for expert, count in enumerate(counts):
        for batch in range((count + 30) // 31):
            slot = first // 31 + expert + batch
            assert slot not in used and slot < shape[0]
            used.add(slot)
        first += count
    assert scales == (shape[0], 32, 128)
    assert shape[1:] == (64, 2048)


@pytest.mark.parametrize("tokens,bits", [(16, 4), (17, 8)])
def test_input_shapes_reject_decode_and_a8(tokens, bits):
    with pytest.raises(ValueError, match="bulk A4"):
        route_input_shapes(FusedGeometry(tokens, 2, 3, 256, 256, 3, 3, bits))


@pytest.mark.parametrize("tokens,bits,packed", [(16, 4, False), (17, 4, True), (640, 4, True), (640, 8, False)])
def test_launch_routes_only_bulk_a4_and_keeps_shared_buffers(tokens, bits, packed):
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device = torch.device("cpu")
    native.activation_bits = bits
    native.fp16_route_workspace = True
    native.weight_lookup = None
    native.configs, native.scratch = {}, {}
    native.pack_kernel, native.gate_kernel, native.down_kernel, native.reduce_kernel = "pack", "gate", "down", "reduce"
    native.route_input_kernel = "routed-input"
    native.route_packed_down = True
    calls = []
    native.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    geometry = FusedGeometry(tokens, 2, 3, 256, 256, 3, 3, bits)
    args = (
        torch.zeros(tokens, 256, dtype=torch.float16),
        None,
        torch.ones(1),
        None,
        torch.ones(1),
        torch.ones(tokens, 2),
        torch.arange(tokens * 2),
        torch.tensor([0, 0, tokens * 2]),
        geometry,
    )
    output = native.grouped(*args)
    assert output.dtype == torch.float32 and output.shape == (tokens, 256)
    gate_index = 2 if packed else 1
    assert calls[gate_index][1][7].shape == native.hidden_code_shape(geometry)
    assert [k for k, _ in calls] == ["pack"] + (["routed-input"] if packed else []) + ["gate", "down"] + (
        ["reduce"] if tokens > 16 else []
    )
    if packed:
        pack_args, gate_args = calls[1][1], calls[2][1]
        assert gate_args[0] is pack_args[0] and gate_args[1] is pack_args[4] and gate_args[2] is pack_args[5]
        if tokens == 640:
            calls.clear()
            native.grouped(*args)
            assert calls[1][1][4] is pack_args[4] and calls[1][1][5] is pack_args[5]
            native.scratch.clear()
            calls.clear()
            native.grouped(*args)
            assert calls[1][1][4] is not pack_args[4]


def test_missing_route_producer_rejected_before_any_partial_kernel_load(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"route_packed_input": True}}))
    with pytest.raises(ValueError, match="compiled producer"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            kernel_factory=lambda *a: pytest.fail("partial kernel load"),
        )


def test_packed_down_requires_shared_wide_input_before_build(tmp_path):
    with pytest.raises(ValueError, match="packed down input requires"):
        build(tmp_path / "build", tmp_path, tmp_path, route_packed_down=True)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("count", [1, 16, 17, 31])
def test_gate_tiles_fill_exact_down_pair_layout_with_zero_tail(count):
    # Independent producer and former down-packing address calculations.
    # Fill a reused slot with sentinel bytes first, then clear each tile's pairs.
    groups, row_bytes, pair_bytes = 64, 32, 2048
    source = torch.arange(count * groups * row_bytes).reshape(count, groups, row_bytes).remainder(127).to(torch.int8)
    packed = torch.full((groups // 2, pair_bytes), 99, dtype=torch.int8)
    owners = set()
    for tile in range(groups // 4):
        packed[tile * 2 : tile * 2 + 2].zero_()
        for row in range(count):
            for group in range(4):
                absolute = tile * 4 + group
                pair, offset = absolute // 2, (absolute % 2 * count + row) * row_bytes
                assert (pair, offset) not in owners
                owners.add((pair, offset))
                packed[pair, offset : offset + row_bytes] = source[row, absolute]
    for pair in range(groups // 2):
        expected = torch.zeros(pair_bytes, dtype=torch.int8)
        for part in range(2):
            for row in range(count):
                offset = (part * count + row) * row_bytes
                expected[offset : offset + row_bytes] = source[row, 2 * pair + part]
        assert torch.equal(packed[pair], expected)


@pytest.mark.parametrize("tokens,bits,packed", [(16, 4, False), (17, 4, True), (640, 4, True), (640, 8, False)])
def test_down_bank_capacity_bypass_and_unchanged_scale_contract(tokens, bits, packed):
    from tools.glm_perf.glm_fused_moe import route_down_shape

    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device = torch.device("cpu")
    native.activation_bits, native.fp16_route_workspace, native.route_packed_down = bits, True, True
    geometry = FusedGeometry(tokens, 8, 72, 4096, 2048, 3, 3, bits)
    scratch = native.allocate_scratch(geometry)
    expected = route_down_shape(geometry) if packed else (tokens * 8, 64, 32)
    assert scratch[3].shape == expected
    assert scratch[4].shape == ((tokens * 8, 64, 32) if bits == 8 else (1,))
    assert scratch[5].shape == (tokens * 8, 64, 8)
    if tokens == 640 and bits == 4:
        assert expected == (238, 32, 2048)


def test_bad_down_provenance_rejected_before_kernel_load(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"route_packed_down": True}}))
    with pytest.raises(ValueError, match="requires routed input packing"):
        NativeFusedMoE(
            tmp_path, namespace="unused", activation_bits=4, kernel_factory=lambda *a: pytest.fail("partial load")
        )


def test_shared_down_capacity_keys_include_expert_count():
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device, native.activation_bits = torch.device("cpu"), 4
    native.fp16_route_workspace, native.route_packed_down = True, True
    native.weight_lookup = None
    native.configs, native.scratch = {}, {}
    native.pack_kernel, native.gate_kernel = "pack", "gate"
    native.down_kernel, native.reduce_kernel = "down", "reduce"
    native.launch = lambda *args: None
    for experts in (1, 72):
        g = FusedGeometry(640, 1, experts, 256, 256, 3, 3, 4)
        native.grouped(
            torch.zeros(640, 256).half(),
            None,
            torch.ones(1),
            None,
            torch.ones(1),
            torch.ones(640, 1),
            torch.arange(640),
            torch.full((experts,), 640),
            g,
        )
    assert len(native.scratch) == 2
    assert sorted(v[3].shape[0] for v in native.scratch.values()) == [22, 93]
