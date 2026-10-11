# SPDX-License-Identifier: Apache-2.0
"""Exact batching and cache-integrity checks for startup weight conversion."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.qwen4exp.benchmark_int8_requant_projection_310 import requantize_weight
from tools.qwen4exp.channel_weight_recipe import channel_bank
from tools.qwen4exp.native_channel_int8 import tile_channel_weights
from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_native_metadata, pack_native_weight


def operands():
    generator = torch.Generator().manual_seed(83)
    packed = torch.randint(-128, 128, (3, 160, 128), dtype=torch.int8, generator=generator)
    scales = (torch.rand(3, 160, 2, generator=generator) * 0.1).half()
    offsets = torch.randint(-4, 5, (3, 160, 2), generator=generator).half()
    bank = SimpleNamespace(
        weight=torch.stack([pack_native_weight(expert)[0] for expert in packed]),
        weight_scale=torch.stack([pack_native_metadata(expert) for expert in scales]),
        weight_offset=torch.stack([pack_native_metadata(expert) for expert in offsets]),
    )
    return bank, packed, scales, offsets


def test_expert_batches_preserve_complete_row_quantization():
    bank, packed, scales, offsets = operands()
    _, codes, expected_scale = requantize_weight(
        packed.flatten(0, 1), scales.flatten(0, 1), offsets.flatten(0, 1).to(torch.int8)
    )
    expected = tile_channel_weights(codes.reshape(3, 160, 256))
    progress = []
    actual, actual_scale, _ = channel_bank(bank, progress=lambda done, total: progress.append((done, total)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(3, 160), rtol=0, atol=0)
    assert progress == [(3, 3)]


def test_cached_bank_reuses_exact_codes_and_rejects_changed_source(tmp_path):
    bank, *_ = operands()
    cache = tmp_path / "bank.safetensors"
    first = channel_bank(bank, cache_path=cache)
    second = channel_bank(bank, cache_path=cache, progress=lambda *_: pytest.fail("cached conversion ran"))
    for actual, expected in zip(second[:2], first[:2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert second[2] == first[2]
    bank.weight[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="source or layout"):
        channel_bank(bank, cache_path=cache)


def test_cached_bank_rejects_corrupted_payload(tmp_path):
    from safetensors import safe_open

    bank, *_ = operands()
    cache = tmp_path / "bank.safetensors"
    channel_bank(bank, cache_path=cache)
    with safe_open(cache, framework="pt") as saved:
        metadata = saved.metadata()
    tensors = load_file(cache)
    tensors["weight"][0, 0, 0] ^= 1
    save_file(tensors, cache, metadata=metadata)
    with pytest.raises(ValueError, match="digest"):
        channel_bank(bank, cache_path=cache)
