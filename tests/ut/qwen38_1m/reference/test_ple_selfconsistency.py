# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the PLE gate + short convolution reference."""

import pytest
import torch

from tests.ut.qwen38_1m.reference.ple_reference import (
    ple_gate,
    ple_grouped_rmsnorm,
    ple_short_conv,
)
from tests.ut.qwen38_1m.reference.tolerances import (
    PLE_CONV_ATOL,
    PLE_CONV_RTOL,
    PLE_GATE_ATOL,
    PLE_GATE_RTOL,
)

_EPS = 1e-6


def _rand(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


@pytest.mark.parametrize("seq_len", [1, 4, 16])
@pytest.mark.parametrize("hc", [1, 2, 3])
@pytest.mark.parametrize("h", [8, 16])
def test_gate_matches_bruteforce_loop(seq_len, hc, h):
    key = _rand((seq_len, hc * h), 1)
    hidden = _rand((seq_len, hc * h), 2)
    value = _rand((seq_len, h), 3)
    nk = _rand((hc * h,), 4)
    nq = _rand((hc * h,), 5)
    ncw = _rand((hc * h,), 6)

    gated, conv_in = ple_gate(key, value, hidden, nk, nq, ncw, _EPS)

    # Independent per-(token, group) brute-force reconstruction.
    gated_bf = torch.empty_like(gated)
    for t in range(seq_len):
        for s in range(hc):
            sl = slice(s * h, (s + 1) * h)
            kk = key[t, sl]
            k_n = kk * torch.rsqrt((kk * kk).mean() + _EPS) * (1.0 + nk[sl])
            hh = hidden[t, sl]
            q_n = hh * torch.rsqrt((hh * hh).mean() + _EPS) * (1.0 + nq[sl])
            d = (k_n * q_n).sum() / (h**0.5)
            sign = torch.sign(d)
            g = torch.sigmoid(sign * torch.sqrt(torch.clamp(d.abs(), min=1e-6)))
            gated_bf[t, sl] = g * value[t]
    torch.testing.assert_close(gated, gated_bf, rtol=PLE_GATE_RTOL, atol=PLE_GATE_ATOL)

    # conv_input is the grouped RMSNorm of the gated output.
    conv_bf = ple_grouped_rmsnorm(gated_bf, ncw, _EPS, h)
    torch.testing.assert_close(conv_in, conv_bf, rtol=PLE_GATE_RTOL, atol=PLE_GATE_ATOL)


def test_gate_in_zero_one():
    key = _rand((10, 32), 7)
    hidden = _rand((10, 32), 8)
    value = _rand((10, 16), 9)
    nk = _rand((32,), 10)
    nq = _rand((32,), 11)
    ncw = _rand((32,), 12)
    gated, _ = ple_gate(key, value, hidden, nk, nq, ncw, _EPS)
    # gated = sigmoid(...) * value, so sign(gated) == sign(value) broadcast.
    hc = 2
    value_rep = value.unsqueeze(1).expand(10, hc, 16).reshape(10, 32)
    assert torch.all(torch.sign(gated) == torch.sign(value_rep))


@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("kernel_size", [2, 3, 4])
def test_short_conv_matches_bruteforce(dilation, kernel_size):
    seq_len, channels = 24, 6
    conv_in = _rand((seq_len, channels), 20)
    gated = _rand((seq_len, channels), 21)
    outer = _rand((seq_len, channels), 22)
    weight = _rand((channels, kernel_size), 23)

    out = ple_short_conv(conv_in, gated, outer, weight, dilation, activation=None)

    # Brute-force causal dilated depthwise conv.
    state_len = (kernel_size - 1) * dilation
    conv_bf = torch.zeros(seq_len, channels, dtype=torch.float64)
    for t in range(seq_len):
        for c in range(channels):
            acc = torch.zeros((), dtype=torch.float64)
            for k in range(kernel_size):
                src = t - state_len + k * dilation
                if src >= 0:
                    acc = acc + weight[c, k] * conv_in[src, c]
            conv_bf[t, c] = acc
    expected = outer + (gated + conv_bf)
    torch.testing.assert_close(out, expected, rtol=PLE_CONV_RTOL, atol=PLE_CONV_ATOL)


def test_short_conv_silu_applied():
    seq_len, channels, kernel_size, dilation = 12, 4, 3, 2
    conv_in = _rand((seq_len, channels), 30)
    gated = torch.zeros(seq_len, channels, dtype=torch.float64)
    outer = torch.zeros(seq_len, channels, dtype=torch.float64)
    weight = _rand((channels, kernel_size), 31)
    linear = ple_short_conv(conv_in, gated, outer, weight, dilation, activation=None)
    silu = ple_short_conv(conv_in, gated, outer, weight, dilation, activation="silu")
    torch.testing.assert_close(silu, linear * torch.sigmoid(linear), rtol=PLE_CONV_RTOL, atol=PLE_CONV_ATOL)


def test_residual_only_when_conv_zero():
    """Zero conv weights -> output is exactly outer_residual + gated."""
    seq_len, channels = 8, 5
    conv_in = _rand((seq_len, channels), 40)
    gated = _rand((seq_len, channels), 41)
    outer = _rand((seq_len, channels), 42)
    weight = torch.zeros(channels, 3, dtype=torch.float64)
    out = ple_short_conv(conv_in, gated, outer, weight, dilation=2, activation="silu")
    torch.testing.assert_close(out, outer + gated, rtol=PLE_CONV_RTOL, atol=PLE_CONV_ATOL)
