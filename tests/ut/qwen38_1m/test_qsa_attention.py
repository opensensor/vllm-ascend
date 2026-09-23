# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity tests for the Ascend 310P Qwen4Exp QSA sparse attention (plan T6.2).

The torch-eager QSA attention (:class:`AscendQwen4ExpQSAAttention` and the
:mod:`vllm_ascend.models.qwen4_exp.ops.qsa_attention` op) must reproduce the T0.6
eager reference (``tests/ut/qwen38_1m/reference/qsa_attention_reference.py``)
within the declared QSA attention tolerance, at boundary context lengths, with
full-block selection and a zero-selection first token. Selection-count
semantics, the output gate, partial rotary (head_dim * 0.25), Q/K GemmaRMSNorm
and run-to-run bitwise stability are all covered.

Parity runs the module/op at float64 accumulation (a float64 dtype policy) so the
kernel and the float64 reference agree at rounding level; the determinism test
runs the pinned float16-main / float32-accum policy path.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import here):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_qsa_attention.py
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.reference.qsa_attention_reference import (
    QSA_HEAD_DIM,
    QSA_NUM_KV_HEADS,
    QSA_NUM_QUERY_HEADS,
    apply_partial_rope,
    dense_masked_gqa_attention,
    gemma_rmsnorm,
    sparse_gqa_attention,
)
from tests.ut.qwen38_1m.reference.tolerances import QSA_ATTN_ATOL, QSA_ATTN_RTOL
from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.ops.qsa_attention import (
    qsa_sparse_gqa_attention,
    qsa_write_kv_to_cache,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_cache import PAD_SLOT_ID
from vllm_ascend.models.qwen4_exp.qsa import (
    AscendQwen4ExpQSAAttention,
    partial_rope_cos_sin,
)

_EPS = 1e-6
_ROPE_THETA = 10_000.0
_PARTIAL_ROTARY_FACTOR = 0.25

# A float64 policy: storage + accumulation in float64 so parity against the
# float64 reference is at rounding level (the pinned policy is float16 main).
_POLICY_F64 = replace(
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    qsa_main_dtype=torch.float64,
    kv_cache_dtype=torch.float64,
    attention_accumulation_dtype=torch.float64,
)


def test_interleaved_mrope_selects_axis_per_frequency_pair():
    positions = torch.tensor([[1, 2], [10, 20], [100, 200]])
    cos, sin = partial_rope_cos_sin(
        positions,
        rotary_dim=8,
        base=_ROPE_THETA,
        dtype=torch.float64,
        mrope_section=[2, 1, 1],
        mrope_interleaved=True,
    )

    inv_freq = 1.0 / (_ROPE_THETA ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    frequency_positions = torch.tensor([[1, 10, 100, 1], [2, 20, 200, 2]], dtype=torch.float32)
    angles = frequency_positions * inv_freq
    expected_cos = torch.cat((torch.cos(angles), torch.cos(angles)), dim=-1).double()
    expected_sin = torch.cat((torch.sin(angles), torch.sin(angles)), dim=-1).double()
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def _rand(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


def _config(num_q_heads, num_kv_heads, head_dim):
    return SimpleNamespace(
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        rope_theta=_ROPE_THETA,
        rms_norm_eps=_EPS,
    )


def _causal_selection(seq_len, context_len, positions, width, seed, *, zero_first=False):
    """Build a per-query causal selection (``-1`` padded) with varying counts."""
    gen = torch.Generator().manual_seed(seed)
    packed = torch.full((seq_len, width), -1, dtype=torch.int64)
    counts = torch.zeros(seq_len, dtype=torch.int64)
    for t in range(seq_len):
        visible = min(int(positions[t].item()) + 1, context_len)
        if zero_first and t == 0:
            continue
        if visible <= 0:
            continue
        n = int(torch.randint(1, min(width, visible) + 1, (1,), generator=gen).item())
        choices = torch.randperm(visible, generator=gen)[:n]
        packed[t, :n] = choices
        counts[t] = n
    return packed, counts


# ---------------------------------------------------------------------------
# Op-level parity: sparse GQA attention == T0.6 reference (float64).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim",
    [(24, 2, 256), (24, 2, 32), (4, 1, 16), (8, 4, 8)],
)
def test_sparse_op_matches_reference(num_q_heads, num_kv_heads, head_dim):
    seq_len, context_len, width = 5, 20, 8
    query = _rand((seq_len, num_q_heads, head_dim), 10)
    key = _rand((context_len, num_kv_heads, head_dim), 11)
    value = _rand((context_len, num_kv_heads, head_dim), 12)
    gate = _rand((seq_len, num_q_heads, head_dim), 13)
    positions = torch.arange(context_len - seq_len, context_len)
    packed, counts = _causal_selection(seq_len, context_len, positions, width, 14)

    got = qsa_sparse_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads, accum_dtype=torch.float64)
    ref = sparse_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads)
    torch.testing.assert_close(got, ref, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


# ---------------------------------------------------------------------------
# Module parity: Q/K norm + partial rope + sparse attention + gate.
# ---------------------------------------------------------------------------
def _module_f64(num_q_heads, num_kv_heads, head_dim, *, qw, kw):
    module = AscendQwen4ExpQSAAttention(
        config=_config(num_q_heads, num_kv_heads, head_dim),
        layer_idx=0,
        dtype_policy=_POLICY_F64,
    )
    with torch.no_grad():
        module.q_norm_weight.copy_(qw)
        module.k_norm_weight.copy_(kw)
    return module


def _reference_pipeline(
    query, key, value, gate, positions, key_positions, packed, counts, qw, kw, num_kv_heads, rotary_dim
):
    q_ref = apply_partial_rope(gemma_rmsnorm(query, qw, _EPS), positions, rotary_dim, _ROPE_THETA)
    k_ref = apply_partial_rope(gemma_rmsnorm(key, kw, _EPS), key_positions, rotary_dim, _ROPE_THETA)
    return sparse_gqa_attention(q_ref, k_ref, value, gate, packed, counts, num_kv_heads)


@pytest.mark.parametrize("context_len", [1, 2, 3, 4, 5, 8, 16])
def test_module_matches_reference_boundary_lengths(context_len):
    num_q_heads, num_kv_heads, head_dim = QSA_NUM_QUERY_HEADS, QSA_NUM_KV_HEADS, 32
    seq_len = min(context_len, 4)
    rotary_dim = int(head_dim * _PARTIAL_ROTARY_FACTOR)
    query = _rand((seq_len, num_q_heads, head_dim), 100 + context_len)
    key = _rand((context_len, num_kv_heads, head_dim), 200 + context_len)
    value = _rand((context_len, num_kv_heads, head_dim), 300 + context_len)
    gate = _rand((seq_len, num_q_heads, head_dim), 400 + context_len)
    qw = _rand((head_dim,), 500 + context_len)
    kw = _rand((head_dim,), 600 + context_len)
    positions = torch.arange(context_len - seq_len, context_len)
    key_positions = torch.arange(context_len)
    packed, counts = _causal_selection(seq_len, context_len, positions, 6, 700 + context_len)

    module = _module_f64(num_q_heads, num_kv_heads, head_dim, qw=qw, kw=kw)
    got = module.forward(
        query, key, value, gate, positions, packed, counts, key_positions=key_positions, accum_dtype=torch.float64
    )
    ref = _reference_pipeline(
        query, key, value, gate, positions, key_positions, packed, counts, qw, kw, num_kv_heads, rotary_dim
    )
    assert got.dtype == torch.float64
    torch.testing.assert_close(got, ref, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_module_real_geometry_parity():
    """Full plan geometry: 24 q-heads, 2 kv-heads, head dim 256, rotary_dim 64."""
    num_q_heads, num_kv_heads, head_dim = QSA_NUM_QUERY_HEADS, QSA_NUM_KV_HEADS, QSA_HEAD_DIM
    rotary_dim = int(head_dim * _PARTIAL_ROTARY_FACTOR)
    assert rotary_dim == 64
    seq_len, context_len = 3, 12
    query = _rand((seq_len, num_q_heads, head_dim), 31)
    key = _rand((context_len, num_kv_heads, head_dim), 32)
    value = _rand((context_len, num_kv_heads, head_dim), 33)
    gate = _rand((seq_len, num_q_heads, head_dim), 34)
    qw = _rand((head_dim,), 35)
    kw = _rand((head_dim,), 36)
    positions = torch.arange(context_len - seq_len, context_len)
    key_positions = torch.arange(context_len)
    packed, counts = _causal_selection(seq_len, context_len, positions, 8, 37)

    module = _module_f64(num_q_heads, num_kv_heads, head_dim, qw=qw, kw=kw)
    assert module.rotary_dim == 64
    got = module.forward(
        query, key, value, gate, positions, packed, counts, key_positions=key_positions, accum_dtype=torch.float64
    )
    ref = _reference_pipeline(
        query, key, value, gate, positions, key_positions, packed, counts, qw, kw, num_kv_heads, rotary_dim
    )
    torch.testing.assert_close(got, ref, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_zero_selection_first_token_is_zero():
    num_q_heads, num_kv_heads, head_dim = 4, 2, 16
    seq_len, context_len = 3, 8
    rotary_dim = int(head_dim * _PARTIAL_ROTARY_FACTOR)
    query = _rand((seq_len, num_q_heads, head_dim), 41)
    key = _rand((context_len, num_kv_heads, head_dim), 42)
    value = _rand((context_len, num_kv_heads, head_dim), 43)
    gate = _rand((seq_len, num_q_heads, head_dim), 44)
    qw = _rand((head_dim,), 45)
    kw = _rand((head_dim,), 46)
    positions = torch.arange(context_len - seq_len, context_len)
    key_positions = torch.arange(context_len)
    packed, counts = _causal_selection(seq_len, context_len, positions, 6, 47, zero_first=True)
    assert int(counts[0].item()) == 0

    module = _module_f64(num_q_heads, num_kv_heads, head_dim, qw=qw, kw=kw)
    got = module.forward(
        query, key, value, gate, positions, packed, counts, key_positions=key_positions, accum_dtype=torch.float64
    )
    ref = _reference_pipeline(
        query, key, value, gate, positions, key_positions, packed, counts, qw, kw, num_kv_heads, rotary_dim
    )
    assert torch.all(got[0] == 0.0)  # zero selection -> zero output (even after gate)
    torch.testing.assert_close(got, ref, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_full_block_selection_equals_dense_masked():
    """Selecting the full causal prefix equals dense attention over that set."""
    num_q_heads, num_kv_heads, head_dim = 8, 4, 16
    seq_len = 4
    context_len = 10
    query = _rand((seq_len, num_q_heads, head_dim), 51)
    key = _rand((context_len, num_kv_heads, head_dim), 52)
    value = _rand((context_len, num_kv_heads, head_dim), 53)
    gate = _rand((seq_len, num_q_heads, head_dim), 54)
    width = context_len
    packed = torch.full((seq_len, width), -1, dtype=torch.int64)
    counts = torch.zeros(seq_len, dtype=torch.int64)
    for t in range(seq_len):
        packed[t, :context_len] = torch.arange(context_len)
        counts[t] = context_len

    got = qsa_sparse_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads, accum_dtype=torch.float64)
    dense = dense_masked_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads)
    torch.testing.assert_close(got, dense, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_selection_count_bounds_attention():
    """Only the first ``valid_count`` packed entries are attended."""
    num_q_heads, num_kv_heads, head_dim = 4, 2, 16
    seq_len, context_len = 3, 12
    query = _rand((seq_len, num_q_heads, head_dim), 61)
    key = _rand((context_len, num_kv_heads, head_dim), 62)
    value = _rand((context_len, num_kv_heads, head_dim), 63)
    gate = _rand((seq_len, num_q_heads, head_dim), 64)
    width = 6
    packed = torch.full((seq_len, width), -1, dtype=torch.int64)
    counts = torch.tensor([2, 3, 1])
    packed[0, :2] = torch.tensor([0, 5])
    packed[1, :3] = torch.tensor([1, 2, 9])
    packed[2, :1] = torch.tensor([7])

    baseline = qsa_sparse_gqa_attention(
        query, key, value, gate, packed, counts, num_kv_heads, accum_dtype=torch.float64
    )
    corrupted = packed.clone()
    corrupted[0, 2:] = 3
    corrupted[1, 3:] = 8
    corrupted[2, 1:] = 4
    after = qsa_sparse_gqa_attention(
        query, key, value, gate, corrupted, counts, num_kv_heads, accum_dtype=torch.float64
    )
    torch.testing.assert_close(after, baseline, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_output_gate_halves_on_zero_gate():
    num_q_heads, num_kv_heads, head_dim = 2, 1, 8
    seq_len, context_len = 2, 6
    query = _rand((seq_len, num_q_heads, head_dim), 71)
    key = _rand((context_len, num_kv_heads, head_dim), 72)
    value = _rand((context_len, num_kv_heads, head_dim), 73)
    packed = torch.full((seq_len, 4), -1, dtype=torch.int64)
    counts = torch.tensor([3, 3])
    packed[:, :3] = torch.tensor([0, 1, 2])

    zero_gate = torch.zeros(seq_len, num_q_heads, head_dim, dtype=torch.float64)
    big_gate = torch.full_like(zero_gate, 1e9)
    ungated = qsa_sparse_gqa_attention(
        query, key, value, zero_gate, packed, counts, num_kv_heads, accum_dtype=torch.float64
    )
    gated = qsa_sparse_gqa_attention(
        query, key, value, big_gate, packed, counts, num_kv_heads, accum_dtype=torch.float64
    )
    torch.testing.assert_close(gated, ungated * 2.0, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


# ---------------------------------------------------------------------------
# Determinism: run-to-run bitwise stability on the pinned float16 policy path.
# ---------------------------------------------------------------------------
def test_run_to_run_bitwise_stable():
    num_q_heads, num_kv_heads, head_dim = QSA_NUM_QUERY_HEADS, QSA_NUM_KV_HEADS, 32
    seq_len, context_len = 4, 16
    module = AscendQwen4ExpQSAAttention(config=_config(num_q_heads, num_kv_heads, head_dim), layer_idx=0)
    assert module.qsa_dtype == torch.float16  # pinned policy
    with torch.no_grad():
        module.q_norm_weight.copy_(_rand((head_dim,), 81).to(module.qsa_dtype))
        module.k_norm_weight.copy_(_rand((head_dim,), 82).to(module.qsa_dtype))
    query = _rand((seq_len, num_q_heads, head_dim), 83).to(torch.float32)
    key = _rand((context_len, num_kv_heads, head_dim), 84).to(torch.float32)
    value = _rand((context_len, num_kv_heads, head_dim), 85).to(torch.float32)
    gate = _rand((seq_len, num_q_heads, head_dim), 86).to(torch.float32)
    positions = torch.arange(context_len - seq_len, context_len)
    key_positions = torch.arange(context_len)
    packed, counts = _causal_selection(seq_len, context_len, positions, 6, 87)

    first = module.forward(query, key, value, gate, positions, packed, counts, key_positions=key_positions)
    second = module.forward(query, key, value, gate, positions, packed, counts, key_positions=key_positions)
    assert first.dtype == torch.float16
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# C8 write hook (Candidate A / task T8.1): reserved seam only.
# ---------------------------------------------------------------------------
def test_c8_write_hook_defaults_unset_and_is_bf16_passthrough():
    module = AscendQwen4ExpQSAAttention(config=_config(4, 2, 8), layer_idx=0)
    assert module.kv_write_quant_hook is None  # C8 hook reserved, not implemented

    key_cache = torch.zeros((4, 8))
    value_cache = torch.zeros((4, 8))
    slots = torch.tensor([2, PAD_SLOT_ID, 0], dtype=torch.long)
    key_rows = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    value_rows = key_rows + 100.0

    module.write_kv_cache(key_cache, value_cache, slots, key_rows, value_rows)
    # BF16/float16 passthrough: rows land verbatim (PAD row skipped).
    assert torch.equal(key_cache[2], key_rows[0])
    assert torch.equal(key_cache[0], key_rows[2])
    assert torch.equal(key_cache[1], torch.zeros(8))
    assert torch.equal(value_cache[2], value_rows[0])


def test_c8_write_hook_routes_rows_when_provided():
    """A provided quant hook is applied to K/V rows before the paged scatter."""
    calls = []

    def spy_hook(k_rows, v_rows):
        calls.append((k_rows.clone(), v_rows.clone()))
        return k_rows * 0 + 7.0, v_rows * 0 + 9.0

    key_cache = torch.zeros((2, 4))
    value_cache = torch.zeros((2, 4))
    slots = torch.tensor([0, 1], dtype=torch.long)
    key_rows = torch.ones((2, 4))
    value_rows = torch.ones((2, 4)) * 2.0

    qsa_write_kv_to_cache(key_cache, value_cache, slots, key_rows, value_rows, quant_hook=spy_hook)
    assert len(calls) == 1
    assert torch.all(key_cache == 7.0)
    assert torch.all(value_cache == 9.0)
