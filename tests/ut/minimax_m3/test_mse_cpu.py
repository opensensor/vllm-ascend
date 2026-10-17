# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving acceleration of the established M3 W4 conversion recipe."""

import importlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.deepseek_w2.w2_format import quantize_weight
from tools.minimax_m3.export_w4 import export_checkpoint, export_tensor
from tools.minimax_m3.probe_w4 import quantize_chunk


@pytest.fixture
def accelerated():
    pytest.importorskip("numba")
    module = importlib.import_module("tools.minimax_m3.mse_cpu")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield module
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["random", "ties", "tiny", "outliers"])
def test_codes_and_float64_scales_match_canonical(accelerated, dtype, kind):
    generator = torch.Generator().manual_seed(33)
    weight = torch.randn(128, 96, generator=generator, dtype=torch.float64)
    if kind == "ties":
        weight = weight.sign()
        weight[:32] = 0
    elif kind == "tiny":
        weight *= 1e-20
    elif kind == "outliers":
        weight[::32, ::32] *= 500
    weight = weight.to(dtype)
    actual = accelerated.quantize_mse_cpu(weight)
    reference = quantize_weight(weight, n_bits=4, method="mse")
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])
    for new, old in zip(quantize_chunk(weight, "mse", mse_backend="numba"), quantize_chunk(weight, "mse")):
        assert torch.equal(new, old)


def test_ambiguous_scales_use_canonical_reduction(accelerated, monkeypatch):
    original = accelerated.compute_block_scales
    calls = []

    def record(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(accelerated, "compute_block_scales", record)
    weight = torch.ones(32, 64)
    actual = accelerated.quantize_mse_cpu(weight)
    expected = quantize_weight(weight, n_bits=4, method="mse")
    assert calls
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("weight", [torch.full((32, 32), torch.inf), torch.zeros(33, 32), torch.zeros(32)])
def test_invalid_input_rejected(accelerated, weight):
    with pytest.raises(ValueError):
        accelerated.quantize_mse_cpu(weight)


def test_export_files_and_resume_are_backend_independent(accelerated, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        f"language_model.model.layers.0.block_sparse_moe.experts.0.{projection}.weight": torch.randn(64, 64).bfloat16()
        for projection in ("w1", "w2", "w3")
    }
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"text_config": {"moe_layer_freq": [1], "num_local_experts": 1}}))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    name = next(iter(tensors))
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
        old = export_tensor(handle.get_slice(name), name, tmp_path / "old.safetensors", method="mse")
        new = export_tensor(
            handle.get_slice(name), name, tmp_path / "new.safetensors", method="mse", mse_backend="numba"
        )
    assert old["sha256"] == new["sha256"]
    target = tmp_path / "export"
    export_checkpoint(source, target, max_tensors=1)
    first = target / "tensor-000000.safetensors"
    modified = first.stat().st_mtime_ns
    assert export_checkpoint(source, target, mse_backend="numba")["complete"]
    assert first.stat().st_mtime_ns == modified
