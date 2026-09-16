# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the GDN reference.

Core acceptance (T0.6): chunked GDN output == unchunked GDN output within the
declared tolerance, across chunk sizes and with/without an initial state.
"""

import pytest
import torch

from tests.ut.qwen38_1m.reference.gdn_reference import (
    causal_depthwise_conv1d,
    gdn_delta_rule_chunked,
    gdn_delta_rule_recurrent,
    gdn_gating,
    preprocess_qk,
)
from tests.ut.qwen38_1m.reference.tolerances import (
    GDN_CHUNK_ATOL,
    GDN_CHUNK_RTOL,
    GDN_CONV_ATOL,
    GDN_CONV_RTOL,
)

# Qwen4Exp GDN test shape (hidden 2560 config uses small head dims; use compact
# but representative head counts/dims for a CPU reference).
_NUM_HEADS = 4
_K_DIM = 16
_V_DIM = 16


def _make_inputs(seq_len: int, seed: int, with_state: bool):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(seq_len, _NUM_HEADS, _K_DIM, generator=gen, dtype=torch.float64)
    k = torch.randn(seq_len, _NUM_HEADS, _K_DIM, generator=gen, dtype=torch.float64)
    v = torch.randn(seq_len, _NUM_HEADS, _V_DIM, generator=gen, dtype=torch.float64)
    a = torch.randn(seq_len, _NUM_HEADS, generator=gen, dtype=torch.float64)
    b = torch.randn(seq_len, _NUM_HEADS, generator=gen, dtype=torch.float64)
    A_log = torch.randn(_NUM_HEADS, generator=gen, dtype=torch.float64)
    dt_bias = torch.randn(_NUM_HEADS, generator=gen, dtype=torch.float64)
    state = None
    if with_state:
        state = torch.randn(_NUM_HEADS, _V_DIM, _K_DIM, generator=gen, dtype=torch.float64)
    g, beta_gate = gdn_gating(a, b, A_log, dt_bias)
    q, k = preprocess_qk(q, k)
    return q, k, v, g, beta_gate, state


@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 64, 65, 130])
@pytest.mark.parametrize("chunk_size", [1, 8, 16, 64])
@pytest.mark.parametrize("with_state", [False, True])
def test_chunked_equals_recurrent(seq_len, chunk_size, with_state):
    q, k, v, g, beta_gate, state = _make_inputs(seq_len, seed=1000 + seq_len + chunk_size, with_state=with_state)
    o_rec, s_rec = gdn_delta_rule_recurrent(q, k, v, g, beta_gate, state)
    o_chunk, s_chunk = gdn_delta_rule_chunked(q, k, v, g, beta_gate, state, chunk_size=chunk_size)
    torch.testing.assert_close(o_chunk, o_rec, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    torch.testing.assert_close(s_chunk, s_rec, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)


def test_chunk_size_invariance():
    """Output must be independent of the chunk size used to compute it."""
    q, k, v, g, beta_gate, state = _make_inputs(97, seed=7, with_state=True)
    reference, _ = gdn_delta_rule_chunked(q, k, v, g, beta_gate, state, chunk_size=97)
    for chunk_size in (1, 3, 16, 32, 64):
        out, _ = gdn_delta_rule_chunked(q, k, v, g, beta_gate, state, chunk_size=chunk_size)
        torch.testing.assert_close(out, reference, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)


def test_gating_decay_is_nonpositive():
    """g = -exp(A_log) * softplus(x) is always <= 0 (a decay)."""
    _, _, _, g, beta_gate, _ = _make_inputs(50, seed=3, with_state=False)
    assert torch.all(g <= 0)
    assert torch.all((beta_gate > 0) & (beta_gate < 1))


def test_causal_conv_matches_manual():
    gen = torch.Generator().manual_seed(11)
    seq_len, channels, kernel = 20, 6, 4
    x = torch.randn(seq_len, channels, generator=gen, dtype=torch.float64)
    weight = torch.randn(channels, kernel, generator=gen, dtype=torch.float64)
    bias = torch.randn(channels, generator=gen, dtype=torch.float64)

    out = causal_depthwise_conv1d(x, weight, bias=bias, activation=None)

    # Brute-force causal reference: y[t,c] = sum_j w[c,j] * x[t - (K-1) + j, c].
    expected = torch.zeros_like(out)
    for t in range(seq_len):
        for c in range(channels):
            acc = bias[c].clone()
            for j in range(kernel):
                src = t - (kernel - 1) + j
                if src >= 0:
                    acc = acc + weight[c, j] * x[src, c]
            expected[t, c] = acc
    torch.testing.assert_close(out, expected, rtol=GDN_CONV_RTOL, atol=GDN_CONV_ATOL)


def test_causal_conv_silu_applied():
    gen = torch.Generator().manual_seed(12)
    x = torch.randn(8, 3, generator=gen, dtype=torch.float64)
    weight = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    linear = causal_depthwise_conv1d(x, weight, activation=None)
    silu = causal_depthwise_conv1d(x, weight, activation="silu")
    torch.testing.assert_close(silu, linear * torch.sigmoid(linear), rtol=GDN_CONV_RTOL, atol=GDN_CONV_ATOL)
