# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the QSA sparse attention reference.

Acceptance (T0.6): partial rotary, Q/K norm, output gate, causal / selection-
count semantics; sparse (gather) attention == dense attention masked to the same
selected set.
"""

import pytest
import torch

from tests.ut.qwen38_1m.reference.qsa_attention_reference import (
    apply_partial_rope,
    dense_masked_gqa_attention,
    gemma_rmsnorm,
    project_qk_norm_rope,
    sparse_gqa_attention,
)
from tests.ut.qwen38_1m.reference.tolerances import QSA_ATTN_ATOL, QSA_ATTN_RTOL

_EPS = 1e-6


def _rand(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


def test_gemma_rmsnorm_matches_manual():
    x = _rand((6, 3, 16), 1)
    w = _rand((16,), 2)
    out = gemma_rmsnorm(x, w, _EPS)
    var = x.square().mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(var + _EPS) * (1.0 + w)
    torch.testing.assert_close(out, expected, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_partial_rope_leaves_tail_untouched():
    x = _rand((5, 2, 16), 3)
    rotary_dim = 8
    positions = torch.arange(5)
    out = apply_partial_rope(x, positions, rotary_dim)
    # Dims >= rotary_dim are passed through unchanged.
    torch.testing.assert_close(out[..., rotary_dim:], x[..., rotary_dim:], rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_rope_position_zero_is_identity():
    x = _rand((1, 4, 16), 4)
    positions = torch.zeros(1, dtype=torch.long)
    out = apply_partial_rope(x, positions, rotary_dim=8)
    torch.testing.assert_close(out, x, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_rope_preserves_norm_of_rotated_part():
    """RoPE is a rotation -> the rotated sub-vector keeps its L2 norm."""
    x = _rand((7, 3, 32), 5)
    rotary_dim = 16
    positions = torch.arange(7) * 3
    out = apply_partial_rope(x, positions, rotary_dim)
    n_in = x[..., :rotary_dim].norm(dim=-1)
    n_out = out[..., :rotary_dim].norm(dim=-1)
    torch.testing.assert_close(n_out, n_in, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [(24, 2, 32), (4, 1, 16), (8, 4, 8)])
def test_sparse_equals_dense_masked(num_q_heads, num_kv_heads, head_dim):
    seq_len = 6
    context_len = 20
    query = _rand((seq_len, num_q_heads, head_dim), 10)
    key = _rand((context_len, num_kv_heads, head_dim), 11)
    value = _rand((context_len, num_kv_heads, head_dim), 12)
    gate = _rand((seq_len, num_q_heads, head_dim), 13)

    # Build a per-query selection (varying counts, -1 padded).
    width = 8
    gen = torch.Generator().manual_seed(14)
    packed = torch.full((seq_len, width), -1, dtype=torch.int64)
    counts = torch.zeros(seq_len, dtype=torch.int64)
    for t in range(seq_len):
        n = int(torch.randint(1, width + 1, (1,), generator=gen).item())
        # pick n distinct causal-ish tokens
        choices = torch.randperm(context_len, generator=gen)[:n]
        packed[t, :n] = choices
        counts[t] = n

    sparse = sparse_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads)
    dense = dense_masked_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads)
    torch.testing.assert_close(sparse, dense, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_selection_count_bounds_attention():
    """Only the first `valid_count` packed entries are attended.

    Corrupting entries beyond the count must not change the output.
    """
    seq_len, num_q_heads, num_kv_heads, head_dim = 3, 4, 2, 16
    context_len = 12
    query = _rand((seq_len, num_q_heads, head_dim), 20)
    key = _rand((context_len, num_kv_heads, head_dim), 21)
    value = _rand((context_len, num_kv_heads, head_dim), 22)
    gate = _rand((seq_len, num_q_heads, head_dim), 23)

    width = 6
    packed = torch.full((seq_len, width), -1, dtype=torch.int64)
    counts = torch.tensor([2, 3, 1])
    packed[0, :2] = torch.tensor([0, 5])
    packed[1, :3] = torch.tensor([1, 2, 9])
    packed[2, :1] = torch.tensor([7])

    baseline = sparse_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads)

    # Fill the region beyond each count with arbitrary valid indices.
    corrupted = packed.clone()
    corrupted[0, 2:] = 3
    corrupted[1, 3:] = 8
    corrupted[2, 1:] = 4
    after = sparse_gqa_attention(query, key, value, gate, corrupted, counts, num_kv_heads)
    torch.testing.assert_close(after, baseline, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_output_gate_applied():
    """Output is attention * sigmoid(gate)."""
    seq_len, num_q_heads, num_kv_heads, head_dim = 2, 2, 1, 8
    context_len = 6
    query = _rand((seq_len, num_q_heads, head_dim), 30)
    key = _rand((context_len, num_kv_heads, head_dim), 31)
    value = _rand((context_len, num_kv_heads, head_dim), 32)
    zero_gate = torch.zeros(seq_len, num_q_heads, head_dim, dtype=torch.float64)
    packed = torch.full((seq_len, 4), -1, dtype=torch.int64)
    counts = torch.tensor([3, 3])
    packed[:, :3] = torch.tensor([0, 1, 2])

    ungated = sparse_gqa_attention(query, key, value, zero_gate, packed, counts, num_kv_heads)
    # sigmoid(0) = 0.5, so a zero gate halves the raw attention output.
    big_gate = torch.full_like(zero_gate, 1e9)
    gated = sparse_gqa_attention(query, key, value, big_gate, packed, counts, num_kv_heads)
    torch.testing.assert_close(gated, ungated * 2.0, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_project_qk_norm_rope_composition():
    """project_qk_norm_rope == gemma_rmsnorm then partial RoPE."""
    seq_len, num_q_heads, num_kv_heads, head_dim = 4, 4, 2, 32
    q = _rand((seq_len, num_q_heads, head_dim), 40)
    k = _rand((seq_len, num_kv_heads, head_dim), 41)
    qn = _rand((head_dim,), 42)
    kn = _rand((head_dim,), 43)
    positions = torch.arange(seq_len)
    rotary_dim = 16

    q_out, k_out = project_qk_norm_rope(q, k, positions, qn, kn, _EPS, rotary_dim)
    q_expected = apply_partial_rope(gemma_rmsnorm(q, qn, _EPS), positions, rotary_dim)
    k_expected = apply_partial_rope(gemma_rmsnorm(k, kn, _EPS), positions, rotary_dim)
    torch.testing.assert_close(q_out, q_expected, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)
    torch.testing.assert_close(k_out, k_expected, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)
