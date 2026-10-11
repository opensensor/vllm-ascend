# SPDX-License-Identifier: Apache-2.0
"""Independent quantized algebra, complete MoE routing, and work-count gates."""

import pytest
import torch

from tools.glm_perf.channel_scale_math import (
    ChannelWeightBank,
    dequantize_weights,
    fused_moe_reference,
    int8_limbs,
    project,
    projection_cost,
    quantize_activations,
    quantize_weights,
    smooth_diagonal,
)
from tools.glm_perf.fused_weight_layout import unpack_cube


@pytest.mark.parametrize("group_size", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("bits", [4, 8])
def test_accumulated_integer_product_equals_independent_int64_matmul(group_size, bits):
    rng = torch.Generator().manual_seed(101 + group_size)
    weights = torch.randn(1, 128, 4096, generator=rng) / 64
    inputs = torch.randn(3, 4096, generator=rng).half()
    smoothing = torch.linspace(0.5, 2, 4096)
    bank = quantize_weights(weights, group_size, smoothing)
    codes = unpack_cube(bank.codes, 4096)[0].long()
    activation, scales = quantize_activations(inputs.float() / smoothing, group_size, bits)
    expected = torch.zeros(3, 128)
    for group in range(4096 // group_size):
        block = slice(group * group_size, (group + 1) * group_size)
        integer = activation[:, group].long() @ codes[:, block].T
        expected += integer.float() * bank.scales[0, :, group] * scales[:, group, None]
    actual = project(inputs, bank, 0, bits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_int8_bias_rows_are_eliminated_by_exact_per_channel_weight_sums():
    values = torch.arange(-127, 128, dtype=torch.int32)
    low, high = int8_limbs(values)
    assert low.min() >= -8 and low.max() <= 7 and high.min() >= -8 and high.max() <= 7
    assert torch.equal(low + 16 * high + 8, values)
    for weight in (-8, -1, 0, 1, 7):
        assert torch.equal(low * weight + 16 * high * weight + 8 * weight, values * weight)


def test_diagonal_smoothing_preserves_unquantized_projection():
    rng = torch.Generator().manual_seed(13)
    weights = torch.randn(2, 128, 256, generator=rng)
    values = torch.randn(4, 256, generator=rng)
    maxima = torch.linspace(0.1, 20, 256)
    diagonal = smooth_diagonal(weights, maxima)
    for expert in range(2):
        torch.testing.assert_close((values / diagonal) @ (weights[expert] * diagonal).T, values @ weights[expert].T)


def test_zero_channels_and_zero_activations_do_not_create_nan():
    bank = quantize_weights(torch.zeros(1, 128, 256))
    activation, scales = quantize_activations(torch.zeros(2, 256), 256, 8)
    assert not activation.any() and torch.equal(scales, torch.ones(2, 1))
    assert not project(torch.zeros(2, 256), bank, 0).any()
    assert torch.equal(smooth_diagonal(torch.zeros(1, 128, 256), torch.zeros(256)), torch.ones(256))


@pytest.mark.parametrize("bits", [4, 8])
def test_complete_moe_matches_explicit_quantized_reference_with_peer_and_zero_routes(bits):
    rng = torch.Generator().manual_seed(71)
    gate = quantize_weights(torch.randn(2, 512, 256, generator=rng) / 32)
    down = quantize_weights(torch.randn(2, 256, 256, generator=rng) / 32)
    x = torch.randn(4, 256, generator=rng).half()
    ids = torch.tensor([[7, 8, 99], [8, 7, 8], [7, 99, 8], [99, 99, 99]])
    weights = torch.tensor([[0.4, 0.6, 0.2], [0.2, 0, 0.7], [0, 0.9, 0.6], [1, 1, 1]], dtype=torch.float32)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for token in range(4):
        for slot in range(3):
            expert = int(ids[token, slot]) - 7
            if not 0 <= expert < 2 or weights[token, slot] == 0:
                continue
            value = project(x[token : token + 1], gate, expert, bits).half().float()
            hidden = (torch.nn.functional.silu(value[:, :256]) * value[:, 256:]).clamp(-65504, 65504).half()
            routed = project(hidden, down, expert, bits).half().float()
            expected[token] += routed[0] * weights[token, slot]
    shared = lambda value: value.float() / 8
    actual = fused_moe_reference(x, gate, down, weights, ids, expert_offset=7, bits=bits, shared=shared)
    torch.testing.assert_close(actual, expected + x.float() / 8, rtol=0, atol=0)


def test_more_precise_reference_and_permanent_correction_sums():
    rng = torch.Generator().manual_seed(1)
    weights = torch.randn(1, 128, 512, generator=rng)
    bank = quantize_weights(weights, 256)
    signed = unpack_cube(bank.codes, 512)
    assert torch.equal(bank.sums, signed.reshape(1, 128, 2, 256).int().sum(-1, dtype=torch.int32))
    assert torch.isfinite(dequantize_weights(bank)).all()
    assert bank.scales.shape == (1, 128, 2)


def test_work_counts_show_removed_epilogues_without_claiming_runtime_speed():
    full = projection_cost(4096, 1, 8, 4096)
    grouped = projection_cost(4096, 1, 8, 256)
    assert full["cube_updates"] == grouped["cube_updates"] == 16
    assert full["product_readbacks"] == 1 and grouped["product_readbacks"] == 16
    assert full["scale_epilogue_elements"] == 128
    assert full["baseline_block32_scale_epilogue_elements"] == 128 * 128
    assert full["integer_product_bound"] < 2**24
    assert "offline" in full["measurement"]


@pytest.mark.parametrize("group", [32, 64, 128, 0, True, 768])
def test_legacy_or_invalid_group_rejected(group):
    with pytest.raises(ValueError):
        quantize_weights(torch.ones(1, 128, 1024), group)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_weight_or_activation_rejected(value):
    with pytest.raises(ValueError):
        quantize_weights(torch.full((1, 128, 256), value))
    with pytest.raises(ValueError):
        quantize_activations(torch.full((1, 256), value), 256, 8)


def test_old_scale_layout_is_not_silently_accepted():
    bank = quantize_weights(torch.ones(1, 128, 256))
    old = ChannelWeightBank(
        bank.codes, bank.scales, bank.sums, bank.smoothing, bank.group_size, layout="cube_n128_k256_v1"
    )
    with pytest.raises(ValueError, match="new channel-scale"):
        old.validate()
