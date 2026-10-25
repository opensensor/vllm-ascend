# SPDX-License-Identifier: Apache-2.0
"""Lossless resident packing and explicit operator-capacity opt-in."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.config.load import LoadConfig

from tools.glm_perf.glm_int4 import unpack_canonical_codes, unpack_nz_w4
from tools.minimax_m3.checkpoint import load_expert_banks, read_manifest
from tools.minimax_m3.export_w4 import export_checkpoint
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts, pack_w4_nz256
from vllm_ascend.model_loader.minimax_m3_w4 import MiniMaxM3W4Loader


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(32, 128), (128, 256), (256, 384)])
def test_nz256_roundtrip_preserves_every_signed_code(shape):
    codes = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(8))
    packed = pack_w4_nz256(codes)
    assert packed.dtype == torch.int8 and packed.nbytes == codes.nbytes
    decoded = unpack_nz_w4(packed[None], shape[1] * 2)[0]
    torch.testing.assert_close(decoded, unpack_canonical_codes(codes, shape[1] * 2), atol=0, rtol=0)


@pytest.mark.parametrize("shape,dtype", [((32, 64), torch.uint8), ((17, 128), torch.uint8), ((32, 128), torch.int8)])
def test_nz256_rejects_incompatible_geometry_or_storage(shape, dtype):
    with pytest.raises(ValueError):
        pack_w4_nz256(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize(
    "options",
    [{"unknown": 1}, {"expert_layout": "nz128"}, {"max_grouped_routes": 129}, {"max_grouped_routes": True}],
)
def test_loader_rejects_unknown_or_unqualified_options(options):
    with pytest.raises(ValueError):
        MiniMaxM3W4Loader(LoadConfig(load_format="minimax_m3_w4", model_loader_extra_config=options))


def test_source_loader_reserves_communication_before_expanding_experts(monkeypatch, tmp_path):
    from vllm_ascend.model_loader import minimax_m3_w4

    events = []
    backbone = type("MiniMaxM3Model", (), {"start_layer": 0, "end_layer": 1})()

    def dense_load(weights):
        assert list(weights) == []
        events.append("dense")
        return set()

    model = SimpleNamespace(
        modules=lambda: iter([backbone]),
        parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="npu"))]),
        named_parameters=lambda: iter([]),
        load_weights=dense_load,
    )
    monkeypatch.setattr(minimax_m3_w4, "read_manifest", lambda folder: {})
    monkeypatch.setattr(minimax_m3_w4, "load_source_banks", lambda *args: events.append("experts"))
    monkeypatch.setattr(minimax_m3_w4, "dense_weights", lambda *args, **kwargs: iter([]))
    monkeypatch.setattr(minimax_m3_w4, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(
        minimax_m3_w4, "get_tp_group", lambda: SimpleNamespace(all_reduce=lambda value: events.append("communication"))
    )
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: events.append("synchronize"), empty_cache=lambda: events.append("release_cache")
        ),
        raising=False,
    )
    loader = MiniMaxM3W4Loader(
        LoadConfig(
            load_format="minimax_m3_w4",
            model_loader_extra_config={
                "expert_layout": "nz256",
                "source_nvfp4": str(tmp_path),
                "source_nvfp4_kernel": str(tmp_path / "kernel.json"),
            },
        )
    )
    loader.load_weights(model, SimpleNamespace(model=str(tmp_path)))
    assert events == [
        "synchronize",
        "release_cache",
        "communication",
        "synchronize",
        "release_cache",
        "experts",
        "dense",
    ]


@pytest.mark.parametrize("route_cap", [512, 2048])
def test_load_time_repack_is_lossless_and_keeps_checkpoint_canonical(tmp_path, route_cap):
    source, target = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    tensors = {
        f"language_model.model.layers.0.block_sparse_moe.experts.{expert}.{projection}.weight": torch.randn(
            256, 256
        ).half()
        for expert in range(4)
        for projection in ("w1", "w2", "w3")
    }
    save_file(tensors, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    (source / "config.json").write_text(json.dumps({"text_config": {"moe_layer_freq": [1], "num_local_experts": 4}}))
    export_checkpoint(source, target, method="minmax")
    manifest = read_manifest(target)
    bank = PackedM3Experts(256, 256, 4, 2, 0, 2, prefix="model.layers.0.experts")
    assert load_expert_banks(bank, target, manifest) == 12
    originals = {name: tensor.clone() for name, tensor in bank.named_buffers()}
    assert load_expert_banks(bank, target, manifest, expert_layout="nz256", max_grouped_routes=route_cap) == 12
    assert bank.max_grouped_routes == route_cap
    for projection in ("w1", "w2", "w3"):
        decoded = unpack_nz_w4(getattr(bank, projection + "_codes"), 256)
        for local in range(2):
            torch.testing.assert_close(
                decoded[local], unpack_canonical_codes(originals[projection + "_codes"][local], 256), atol=0, rtol=0
            )
        torch.testing.assert_close(getattr(bank, projection + "_scales"), originals[projection + "_scales"])
    with pytest.raises(ValueError, match="matching native"):
        bank(torch.ones(1, 256).half(), torch.ones(1, 4))
    assert load_expert_banks(bank, target, manifest) == 12
    assert bank.max_grouped_routes == 128
    for name, tensor in bank.named_buffers():
        torch.testing.assert_close(tensor, originals[name], atol=0, rtol=0)
