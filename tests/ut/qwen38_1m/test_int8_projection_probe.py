# SPDX-License-Identifier: Apache-2.0
"""Independent math and layout gates for the isolated INT8 experiment."""

from io import BytesIO

import pytest
import torch

from tools.qwen4exp.benchmark_int8_projection_probe_310 import (
    ordered_reference,
    prepare_activation,
    prepare_weights,
    server_activity,
)
from vllm_ascend.models.qwen4_exp.w4a8_int4 import quantize_activation_limbs


def unpack(packed):
    values = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).flatten(-2)
    return torch.where(values >= 8, values - 16, values).int()


@pytest.mark.parametrize("inputs", [128, 640, 2560])
@pytest.mark.parametrize("rows", [1, 3, 17])
def test_direct_quantization_matches_qualified_limbs(inputs, rows):
    x = torch.randn(rows, inputs, generator=torch.Generator().manual_seed(17)).half()
    x[0, :128].zero_()
    x[-1, -1] = torch.finfo(torch.float16).smallest_normal
    low, high, scales, sums = quantize_activation_limbs(x)
    quantized, direct_scales = prepare_activation(x)
    assert torch.equal(quantized.int(), unpack(low) + 16 * unpack(high) + 8)
    assert torch.equal(direct_scales[..., 0], scales)
    assert torch.equal(quantized.reshape(rows, -1, 128).int().sum(-1).float(), sums)


@pytest.mark.parametrize("inputs,outputs", [(128, 160), (640, 2560), (2560, 640)])
def test_expansion_preserves_signed_codes_offsets_and_group_layout(inputs, outputs):
    generator = torch.Generator().manual_seed(37)
    packed = torch.randint(-128, 128, (outputs, inputs // 2), dtype=torch.int8, generator=generator)
    offsets = torch.randint(-8, 8, (outputs, inputs // 128), dtype=torch.int8, generator=generator)
    scales = torch.rand(offsets.shape, generator=generator).half()
    centered, tiles, tile_scales = prepare_weights(packed, scales, offsets)
    expected = unpack(packed).reshape(outputs, -1, 128) - offsets.int().unsqueeze(-1)
    assert torch.equal(centered.reshape_as(expected).int(), expected)
    restored = tiles.permute(0, 3, 1, 2, 4).reshape(outputs, inputs)
    assert torch.equal(restored, centered)
    assert torch.equal(tile_scales.transpose(1, 2).reshape_as(scales), scales)
    assert tiles.numel() == packed.numel() * 2


def test_ordered_int8_projection_matches_two_limb_correction():
    generator = torch.Generator().manual_seed(83)
    x = torch.randn(3, 640, generator=generator).half()
    packed = torch.randint(-128, 128, (160, 320), dtype=torch.int8, generator=generator)
    offsets = torch.randint(-8, 8, (160, 5), dtype=torch.int8, generator=generator)
    scales = (torch.rand(160, 5, generator=generator) * 0.01).half()
    centered, _, _ = prepare_weights(packed, scales, offsets)
    quantized, xs = prepare_activation(x)
    actual = ordered_reference(quantized, centered, scales, xs)
    low, high, _, sums = quantize_activation_limbs(x)
    codes = unpack(packed)
    result = torch.zeros(3, 160)
    for group in range(5):
        part = slice(group * 128, (group + 1) * 128)
        weight = codes[:, part]
        product = (unpack(low)[:, part] @ weight.t()).float()
        product += 16 * (unpack(high)[:, part] @ weight.t()).float()
        product -= offsets[:, group].float().unsqueeze(0) * sums[:, group].unsqueeze(1)
        product += 8 * weight.sum(-1).float().unsqueeze(0)
        product *= scales[:, group].float().unsqueeze(0)
        product *= xs[:, group, 0].unsqueeze(1)
        result += product
    assert torch.equal(actual, result.half())


def test_rejects_overflowing_offset():
    packed = torch.zeros((160, 64), dtype=torch.int8)
    scales = torch.ones((160, 1), dtype=torch.float16)
    with pytest.raises(ValueError, match="fit lossless"):
        prepare_weights(packed, scales, torch.full((160, 1), -128, dtype=torch.int8))


def test_activity_guard_aggregates_active_and_completed_requests(monkeypatch):
    payload = b"""# HELP ignored
vllm:num_requests_running{engine="0"} 1
vllm:num_requests_waiting{engine="0"} 2
vllm:num_requests_waiting_by_reason{reason="capacity"} 2
vllm:request_success_total{reason="stop"} 17
vllm:request_success_total{reason="length"} 3
"""
    monkeypatch.setattr(
        "tools.qwen4exp.benchmark_int8_projection_probe_310.urlopen", lambda *args, **kwargs: BytesIO(payload)
    )
    assert server_activity("http://127.0.0.1:8001") == (3, 20)


def test_activity_guard_fails_closed_when_metrics_are_missing(monkeypatch):
    monkeypatch.setattr(
        "tools.qwen4exp.benchmark_int8_projection_probe_310.urlopen",
        lambda *args, **kwargs: BytesIO(b"# unavailable\n"),
    )
    with pytest.raises(RuntimeError, match="required timing guard"):
        server_activity("http://127.0.0.1:8001")
