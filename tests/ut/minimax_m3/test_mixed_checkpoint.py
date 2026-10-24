# SPDX-License-Identifier: Apache-2.0
"""CPU gates for the M3.1 source format and bounded download contract."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.minimax_m3.checkpoint import read_manifest
from tools.minimax_m3.export_mixed_w4 import export_checkpoint
from tools.minimax_m3.export_w4 import mixed_weight_groups
from tools.minimax_m3.layer_probe import selection
from tools.minimax_m3.mixed_checkpoint import dequant_mxfp8, dequant_nvfp4, fetch_range
from tools.minimax_m3.probe_w4 import quantize_chunk
from vllm_ascend._310p.ops.minimax_m3 import unpack_w4


def test_nvfp4_all_codes_order_sign_and_inverse_global_scale():
    packed = torch.tensor([[16 * (i + 1) + i for i in range(0, 16, 2)]], dtype=torch.uint8)
    scales = torch.tensor([[4.0]], dtype=torch.float8_e4m3fn)
    decoded = dequant_nvfp4(packed, scales, torch.tensor(2.0))
    expected = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]]) * 2
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)


@pytest.mark.parametrize("global_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_nvfp4_rejects_invalid_global_scale(global_scale):
    with pytest.raises(ValueError, match="global scale"):
        dequant_nvfp4(
            torch.zeros(1, 8, dtype=torch.uint8), torch.ones(1, 1).to(torch.float8_e4m3fn), torch.tensor(global_scale)
        )


def test_nvfp4_group_scales_and_w4_requantization():
    packed = torch.full((32, 16), 0x71, dtype=torch.uint8)
    scales = torch.tensor([[1.0, 2.0]]).repeat(32, 1).to(torch.float8_e4m3fn)
    weight = dequant_nvfp4(packed, scales, torch.tensor(8.0))
    assert weight[0, 0] == 0.0625 and weight[0, 1] == 0.75
    assert weight[0, 16] == 0.125 and weight[0, 17] == 1.5
    codes, block_scales, reconstructed = quantize_chunk(weight, "minmax")
    torch.testing.assert_close(unpack_w4(codes, block_scales), reconstructed, atol=0, rtol=0)
    assert not torch.equal(codes, packed)


def test_mxfp8_uses_exponent_bytes():
    weight = torch.ones(2, 64).to(torch.float8_e4m3fn)
    scales = torch.tensor([[126, 128], [127, 125]], dtype=torch.uint8)
    actual = dequant_mxfp8(weight, scales)
    expected = torch.tensor([[0.5, 2.0], [1.0, 0.25]]).repeat_interleave(32, -1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["nan_scale", "wrong_shape", "wrong_dtype", "nan_weight"])
def test_mxfp8_rejects_invalid_source(kind):
    weight = torch.ones(2, 32).to(torch.float8_e4m3fn)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    if kind == "nan_scale":
        scales[0, 0] = 255
    elif kind == "wrong_shape":
        scales = scales.T
    elif kind == "wrong_dtype":
        scales = scales.float()
    else:
        weight = torch.full((2, 32), float("nan")).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequant_mxfp8(weight, scales)


def test_range_rejects_full_download_before_reading(monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        @property
        def raw(self):
            raise AssertionError("Must not read a full multi-GB shard")

    monkeypatch.setattr("tools.minimax_m3.mixed_checkpoint.requests.get", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="exact byte range"):
        fetch_range("https://example.invalid/shard", 0, 7)


def test_sparse_selection_causal_partial_block_and_local_forcing():
    # The first query cannot see the enormous value in the following block.
    q = torch.ones(2, 1, 1)
    k = torch.tensor([1.0, 2.0, 3.0, 4.0, 1000.0]).view(5, 1, 1)
    _, ids = selection(q, k, length=5, topk=1, block_size=2)
    assert ids.tolist() == [[[1]], [[2]]]


def test_sparse_selection_is_per_head():
    q = torch.tensor([[[1.0], [-1.0]]])
    k = torch.tensor([-9.0, -8.0, 7.0, 8.0, 0.0]).view(5, 1, 1)
    _, ids = selection(q, k, length=5, topk=2, block_size=2)
    assert ids.tolist() == [[[2, 1], [2, 0]]]


@pytest.fixture
def mixed_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weights = {}
    for projection in ("w1", "w2", "w3"):
        base = f"language_model.model.layers.0.block_sparse_moe.experts.0.{projection}"
        weights[base + ".weight_packed"] = torch.full((32, 16), 0x71, dtype=torch.uint8)
        weights[base + ".weight_scale"] = torch.ones(32, 2).to(torch.float8_e4m3fn)
        weights[base + ".weight_global_scale"] = torch.tensor(4.0)
        weights[base + ".input_global_scale"] = torch.tensor(32.0)
    base = "language_model.model.layers.0.self_attn.q_proj"
    weights[base + ".weight"] = torch.ones(32, 32).to(torch.float8_e4m3fn)
    weights[base + ".weight_scale_inv"] = torch.full((32, 1), 126, dtype=torch.uint8)
    weights["language_model.model.layers.0.input_layernorm.weight"] = torch.ones(32).bfloat16()
    save_file(weights, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )
    config = {
        "text_config": {"moe_layer_freq": [1], "num_local_experts": 1, "hidden_size": 32, "intermediate_size": 32},
        "quantization_config": {
            "quant_method": "mxfp8",
            "moe_quant_format": "nvfp4-pack-quantized",
            "group_size": 16,
            "weight_block_size": [1, 32],
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    return source


def test_mixed_export_resumes_and_removes_source_quantization(mixed_source, tmp_path):
    dest = tmp_path / "converted"
    source_bytes = (mixed_source / "config.json").read_bytes()
    assert not export_checkpoint(mixed_source, dest, max_tensors=2, mse_backend="torch")["complete"]
    assert not (dest / "minimax_m3_w4.json").exists()
    result = export_checkpoint(mixed_source, dest, mse_backend="torch")
    assert result == {"complete": True, "tensor_count": 5}
    manifest = read_manifest(dest)
    converted = json.loads((dest / "config.json").read_text())
    assert "quantization_config" not in converted
    assert (mixed_source / "config.json").read_bytes() == source_bytes
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    payload = load_file(dest / manifest["weights"][name]["file"])[name]
    assert payload.dtype == torch.float16
    torch.testing.assert_close(payload, torch.full((32, 32), 0.5).half(), atol=0, rtol=0)
    assert export_checkpoint(mixed_source, dest, mse_backend="torch") == result


def test_mixed_export_rejects_corrupt_resume(mixed_source, tmp_path):
    dest = tmp_path / "converted"
    export_checkpoint(mixed_source, dest, max_tensors=1, mse_backend="torch")
    (dest / "tensor-000000.safetensors").write_bytes(b"broken")
    with pytest.raises(ValueError, match="Corrupt"):
        export_checkpoint(mixed_source, dest, mse_backend="torch")


@pytest.mark.parametrize("missing", ["weight_scale", "weight_global_scale", "input_global_scale"])
def test_mixed_groups_reject_missing_companions(mixed_source, missing):
    weight_map = json.loads((mixed_source / "model.safetensors.index.json").read_text())["weight_map"]
    del weight_map[f"language_model.model.layers.0.block_sparse_moe.experts.0.w1.{missing}"]
    with pytest.raises(ValueError, match="companions"):
        mixed_weight_groups(weight_map)


def test_loader_rejects_removed_companion_in_source_index(mixed_source, tmp_path):
    dest = tmp_path / "converted"
    export_checkpoint(mixed_source, dest, mse_backend="torch")
    path = dest / "source_index.json"
    index = json.loads(path.read_text())
    del index["weight_map"]["language_model.model.layers.0.block_sparse_moe.experts.0.w1.input_global_scale"]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="source index digest"):
        read_manifest(dest)
