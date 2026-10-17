# SPDX-License-Identifier: Apache-2.0
"""CPU tests: python -m pytest --noconftest tests/ut/minimax_m3 -q."""

import json
import struct
from pathlib import Path

import pytest
import torch

from tools.minimax_m3.audit_checkpoint import INDEX_FILE, audit_checkpoint, read_header
from tools.minimax_m3.probe_w4 import probe_tensor, quantize_chunk


def write_shard(path: Path, tensors: dict) -> None:
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    size = max(info["data_offsets"][1] for info in tensors.values())
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(size))


@pytest.fixture
def checkpoint(tmp_path):
    text = {
        "num_hidden_layers": 4,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "moe_layer_freq": [0, 1, 1, 1],
        "hidden_act": "swigluoai",
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "max_position_embeddings": 1024,
        "sparse_attention_config": {
            "sparse_num_index_heads": 2,
            "sparse_index_dim": 4,
            "sparse_attention_freq": [0, 1, 1, 1],
        },
    }
    header = {}
    cursor = 0
    for layer in range(1, 4):
        for expert in range(2):
            for projection in ("w1", "w2", "w3"):
                name = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"
                header[name] = {"dtype": "BF16", "shape": [32, 32], "data_offsets": [cursor, cursor + 2048]}
                cursor += 2048
    name = "language_model.model.layers.0.input_layernorm.weight"
    header[name] = {"dtype": "F32", "shape": [32], "data_offsets": [cursor, cursor + 128]}
    write_shard(tmp_path / "model.safetensors", header)
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["MiniMaxM3SparseForConditionalGeneration"], "text_config": text})
    )
    (tmp_path / INDEX_FILE).write_text(
        json.dumps({"metadata": {"total_size": cursor + 128}, "weight_map": dict.fromkeys(header, "model.safetensors")})
    )
    return tmp_path


def test_audit_accounts_scales_fp32_and_cache_replication(checkpoint):
    result = audit_checkpoint(checkpoint)
    assert result["candidate"]["unique_weight_bytes"] == 18 * (512 + 2) + 128
    assert sum(result["candidate"]["pp_stage_unique_bytes"]) == 18 * 514 + 128
    assert result["candidate"]["tp2_fp16_cache_bytes_per_token_all_stages"] == 4 * 2 * 2 * 4 * 2 + 3 * 4 * 2 * 2
    assert result["candidate"]["tp2_pp3_dimension_candidate"]
    assert not result["candidate"]["tp6_query_heads_divisible"]


def test_stale_index_size_is_reported_without_changing_source(checkpoint):
    path = checkpoint / INDEX_FILE
    index = json.loads(path.read_text())
    index["metadata"]["total_size"] += 1024
    path.write_text(json.dumps(index))
    before = path.read_bytes()
    result = audit_checkpoint(checkpoint)
    assert "Index total_size" in result["warnings"][0]
    assert path.read_bytes() == before


def test_missing_shard_fails(checkpoint):
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        audit_checkpoint(checkpoint)


def test_unindexed_tensor_fails(checkpoint):
    path = checkpoint / INDEX_FILE
    index = json.loads(path.read_text())
    index["weight_map"].pop(next(iter(index["weight_map"])))
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="mapping mismatch"):
        audit_checkpoint(checkpoint)


def test_incomplete_expert_bank_fails(checkpoint):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["text_config"]["num_local_experts"] = 3
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="coverage"):
        audit_checkpoint(checkpoint)


@pytest.mark.parametrize("content", [b"short", struct.pack("<Q", 1 << 30), struct.pack("<Q", 32) + b"{}"])
def test_invalid_header_fails(tmp_path, content):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_header(path)


def test_truncated_payload_fails(checkpoint):
    path = checkpoint / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="offsets/size"):
        audit_checkpoint(checkpoint)


def test_signed_code_grid_uses_stored_scale_and_handles_zero_blocks():
    values = torch.arange(-8, 8, dtype=torch.float32).repeat(64).reshape(32, 32)
    packed, scales, reconstructed = quantize_chunk(values, "minmax")
    assert packed.shape == (32, 16)
    assert scales.dtype == torch.float16
    # Existing quantizer uses half-step cover: max(7 / 7.5, 8 / 8.5).
    # Integer codes survive exactly, but reconstruction uses rounded scales.
    expected_scale = torch.tensor(8 / 8.5, dtype=torch.float16).float()
    torch.testing.assert_close(reconstructed, values * expected_scale, rtol=0, atol=0)
    _, scales, reconstructed = quantize_chunk(torch.zeros(32, 32), "minmax")
    assert (scales > 0).all()
    assert torch.count_nonzero(reconstructed) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e10])
def test_unrepresentable_source_fails(value):
    with pytest.raises(ValueError):
        quantize_chunk(torch.full((32, 32), value), "minmax")


def test_real_file_probe_hashes_zero_tensor(checkpoint):
    weight_map = json.loads((checkpoint / INDEX_FILE).read_text())["weight_map"]
    name = next(iter(weight_map))
    report = probe_tensor(checkpoint, weight_map, name, 32, "minmax")
    assert report["packed_bytes"] == 512
    assert report["scale_bytes"] == 2
    assert report["weight_nrmse"] == 0
    assert report["weight_cosine"] is None
    assert len(report["source_tensor_sha256"]) == 64


def test_row_chunks_preserve_packed_format():
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(64, 64, generator=generator).to(torch.bfloat16)
    whole = quantize_chunk(weight, "mse")
    first = quantize_chunk(weight[:32], "mse")
    second = quantize_chunk(weight[32:], "mse")
    for expected, left, right in zip(whole, first, second):
        torch.testing.assert_close(torch.cat((left, right)), expected, rtol=0, atol=0)
