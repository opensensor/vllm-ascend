# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Candidate A C8 (signed-INT8) main QSA K/V cache
(plan T8.1).

Everything runs host-side with NO NPU and NO Triton: the C8 quant/dequant, the
fused write/read paths (wired through T6.2's ``qsa_write_kv_to_cache`` quant
hook), the accuracy harness (C8-vs-BF16 selection sets + short-context logits),
and the additive C8 main-K/V spec in ``kv_cache``.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_qsa_c8.py

Acceptance (T8.1):
  * Round-trip identity within the pre-declared tolerance (INT8 QDQ of K/V):
    |err| <= scale/2 + C8_ROUNDTRIP_EPS per element.
  * Selection-set agreement (C8 vs BF16) on T0.4-seeded short probes >=
    C8_SELECTION_AGREEMENT_BAR (0.90); short-context attention-logit relative
    error < C8_LOGITS_REL (0.05).
  * The C8 main-K/V spec materializes; its bytes are exactly half BF16, and the
    1M byte math reproduces the PRD §6 footprint.
"""

import sys

import pytest
import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.qwen38_1m.reference.qsa_indexer_reference import (
    qsa_select_tokens,
    selected_token_set,
)
from tools.qwen38_1m.corpus_gen import make_records
from vllm_ascend.models.qwen4_exp import kv_cache as kvc
from vllm_ascend.models.qwen4_exp import qsa_c8
from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY as POLICY,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_attention import qsa_sparse_gqa_attention

_MIB = 1024 * 1024
_GIB = 1024 * _MIB


# =========================================================================
# Host-safety
# =========================================================================
def test_c8_module_import_does_not_pull_torch_npu():
    # The 310P host lane has no NPU; the C8 module must stay host-safe.
    assert "torch_npu" not in sys.modules
    # The C8 payload dtype is signed INT8 (1 byte), not the e4m3 indexer C8.
    assert qsa_c8.C8QSAMainKVCache  # importable
    assert kvc.QSA_C8_MAIN_DTYPE is torch.int8


# =========================================================================
# Round-trip identity (INT8 QDQ)
# =========================================================================
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "num_tokens,num_kv_heads,head_dim",
    [(1, 1, 128), (5, 2, 256), (17, 2, 64), (33, 4, 128)],
)
def test_kv_int8_roundtrip_within_scale_half(seed, num_tokens, num_kv_heads, head_dim):
    gen = torch.Generator().manual_seed(seed)
    rows = torch.randn(num_tokens, num_kv_heads * head_dim, generator=gen) * 3.0
    q, scale = qsa_c8.quantize_kv_int8(rows, num_kv_heads)
    assert q.dtype == torch.int8
    assert torch.all(q >= qsa_c8.INT8_MIN) and torch.all(q <= qsa_c8.INT8_MAX)
    # Per-(token, head) scale: one scale per (token, kv head).
    assert scale.shape == (num_tokens, num_kv_heads)
    rec = qsa_c8.dequantize_kv_int8(q, scale, num_kv_heads)
    # Hard, correct bound: |err| <= scale/2 (half a grid step) per element.
    scale_full = scale.repeat_interleave(head_dim, dim=1)
    err = (rec - rows).abs()
    assert torch.all(err <= scale_full / 2 + qsa_c8.C8_ROUNDTRIP_EPS)


def test_scale_is_per_token_per_head():
    """Each (token, head) gets its own amax scale; heads do not share a scale."""
    # 1 token, 2 heads of dim 2 with very different magnitudes.
    rows = torch.tensor([[1.0, -2.0, 100.0, -50.0]])  # head0 amax 2, head1 amax 100
    _, scale = qsa_c8.quantize_kv_int8(rows, num_kv_heads=2)
    assert scale.shape == (1, 2)
    torch.testing.assert_close(scale[0, 0], torch.tensor(2.0 / 127.0))
    torch.testing.assert_close(scale[0, 1], torch.tensor(100.0 / 127.0))


def test_zero_row_is_safe():
    rows = torch.zeros(3, 2 * 16)
    q, scale = qsa_c8.quantize_kv_int8(rows, num_kv_heads=2)
    assert torch.all(q == 0)
    assert torch.all(qsa_c8.dequantize_kv_int8(q, scale, 2) == 0)


def test_flat_and_head_layouts_agree():
    gen = torch.Generator().manual_seed(9)
    flat = torch.randn(6, 2 * 32, generator=gen)
    heads = flat.view(6, 2, 32)
    q_flat, s_flat = qsa_c8.quantize_kv_int8(flat, 2)
    q_heads, s_heads = qsa_c8.quantize_kv_int8(heads, 2)
    assert q_flat.shape == (6, 64) and q_heads.shape == (6, 2, 32)
    torch.testing.assert_close(s_flat, s_heads)
    torch.testing.assert_close(q_flat.view(6, 2, 32).to(torch.int32), q_heads.to(torch.int32))


# =========================================================================
# Fused write/read paths through the T6.2 quant hook
# =========================================================================
def test_fused_write_read_through_t62_hook():
    """The cache writes INT8 via qsa_write_kv_to_cache(quant_hook=...) and reads
    back within the round-trip bound; PAD slots are skipped."""
    num_slots, num_kv_heads, head_dim, num_tokens = 32, 2, 64, 5
    cache = qsa_c8.C8QSAMainKVCache(num_slots, num_kv_heads, head_dim)
    gen = torch.Generator().manual_seed(3)
    width = num_kv_heads * head_dim
    key_rows = torch.randn(num_tokens, width, generator=gen)
    value_rows = torch.randn(num_tokens, width, generator=gen)
    slot_mapping = torch.tensor([2, 5, 7, qsa_c8.PAD_SLOT_ID, 9])

    cache.write(slot_mapping, key_rows, value_rows)

    # Payload landed in int8; PAD slot 0 (and untouched slots) stay zero.
    assert cache.key_int8.dtype == torch.int8
    assert torch.all(cache.key_int8[0] == 0)

    # Read back the written (non-PAD) slots and check the round-trip bound.
    written = torch.tensor([2, 5, 7, 9])
    k_back = cache.gather_key_rows(written).reshape(4, width)
    v_back = cache.gather_value_rows(written).reshape(4, width)
    src_rows = torch.tensor([0, 1, 2, 4])  # token rows whose slots were valid
    _, k_scale = qsa_c8.quantize_kv_int8(key_rows[src_rows], num_kv_heads)
    k_scale_full = k_scale.repeat_interleave(head_dim, dim=1)
    assert torch.all((k_back - key_rows[src_rows]).abs() <= k_scale_full / 2 + qsa_c8.C8_ROUNDTRIP_EPS)
    _, v_scale = qsa_c8.quantize_kv_int8(value_rows[src_rows], num_kv_heads)
    v_scale_full = v_scale.repeat_interleave(head_dim, dim=1)
    assert torch.all((v_back - value_rows[src_rows]).abs() <= v_scale_full / 2 + qsa_c8.C8_ROUNDTRIP_EPS)


def test_quant_hook_records_per_head_scales():
    recorder: dict = {}
    hook = qsa_c8.make_c8_kv_quant_hook(num_kv_heads=2, recorder=recorder)
    gen = torch.Generator().manual_seed(1)
    k = torch.randn(4, 2 * 8, generator=gen)
    v = torch.randn(4, 2 * 8, generator=gen)
    kq, vq = hook(k, v)
    assert kq.dtype == torch.int8 and vq.dtype == torch.int8
    assert recorder["key_scale"].shape == (4, 2)
    assert recorder["value_scale"].shape == (4, 2)


# =========================================================================
# Accuracy harness: T0.4-seeded selection-set agreement (C8 vs BF16 keys)
# =========================================================================
# Indexer probe geometry (T6.1 heads/dim/ratio). A small token budget forces
# top-k *pruning* on a short context so C8 vs BF16 selection can actually differ
# (a full-prefix dense selection would agree trivially).
_PROBE_N_HEADS = 4
_PROBE_HEAD_DIM = 128
_PROBE_RATIO = 4
_PROBE_TOKEN_TOPK = 64  # block_topk = 16
_PROBE_CONTEXT = 256  # 64 visible blocks -> pruned to 16
_PROBE_QUERIES = 8


def _t04_probe_seed() -> int:
    """Derive a deterministic seed from the T0.4 corpus records (grounds the
    probe in the real retrieval corpus without needing a tokenizer or 1M tokens)."""
    records = make_records()
    text = "".join(r.render() for r in records)
    return sum(ord(c) for c in text) & 0x7FFFFFFF


def _build_probe(seed_offset: int):
    seed = (_t04_probe_seed() + seed_offset) & 0x7FFFFFFF
    gk = torch.Generator().manual_seed(seed)
    raw_keys = torch.randn(_PROBE_CONTEXT, _PROBE_HEAD_DIM, generator=gk)
    gq = torch.Generator().manual_seed(seed ^ 0x5A5A5A5A)
    queries = torch.randn(_PROBE_QUERIES, _PROBE_N_HEADS, _PROBE_HEAD_DIM, generator=gq)
    positions = torch.arange(_PROBE_CONTEXT - _PROBE_QUERIES, _PROBE_CONTEXT, dtype=torch.int64)
    return raw_keys, queries, positions


def _selection_sets(raw_keys, queries, positions):
    packed, _ = qsa_select_tokens(
        queries, raw_keys, positions, compress_ratio=_PROBE_RATIO, token_topk=_PROBE_TOKEN_TOPK
    )
    return [selected_token_set(packed[t]) for t in range(packed.shape[0])]


def test_c8_selection_set_pruning_is_active():
    """Guard the probe: the budget must actually prune (else agreement is trivial)."""
    raw_keys, queries, positions = _build_probe(0)
    packed, counts = qsa_select_tokens(
        queries, raw_keys, positions, compress_ratio=_PROBE_RATIO, token_topk=_PROBE_TOKEN_TOPK
    )
    # Selected token count is bounded by the token budget (+ causal tail), well
    # below the full context, proving top-k pruning is exercised.
    assert int(counts.max().item()) <= _PROBE_TOKEN_TOPK + _PROBE_RATIO - 1
    assert int(counts.max().item()) < _PROBE_CONTEXT


@pytest.mark.parametrize("seed_offset", [0, 1, 2, 3, 4])
def test_c8_selection_agreement_above_bar(seed_offset):
    raw_keys, queries, positions = _build_probe(seed_offset)
    bf16_sets = _selection_sets(raw_keys, queries, positions)
    # C8-quantize the keys the indexer scores (single kv head), dequant, reselect.
    c8_keys, _ = qsa_c8.roundtrip_kv_int8(raw_keys, num_kv_heads=1)
    c8_sets = _selection_sets(c8_keys, queries, positions)
    agreement = qsa_c8.selection_set_agreement(bf16_sets, c8_sets)
    assert agreement["mean_retained"] >= qsa_c8.C8_SELECTION_AGREEMENT_BAR


def test_c8_quant_actually_perturbs_but_agreement_holds():
    """Sanity: C8 keys are not identical to BF16 (quant is doing something), yet
    selection agreement stays above the bar across probes."""
    raw_keys, queries, positions = _build_probe(7)
    c8_keys, scale = qsa_c8.roundtrip_kv_int8(raw_keys, num_kv_heads=1)
    assert scale.shape == (_PROBE_CONTEXT, 1)
    assert not torch.allclose(c8_keys, raw_keys)  # real quantization noise
    agreement = qsa_c8.selection_set_agreement(
        _selection_sets(raw_keys, queries, positions),
        _selection_sets(c8_keys, queries, positions),
    )
    assert agreement["mean_retained"] >= qsa_c8.C8_SELECTION_AGREEMENT_BAR
    assert agreement["mean_jaccard"] > 0.5


# =========================================================================
# Accuracy harness: short-context attention logits (C8 K/V vs BF16 K/V)
# =========================================================================
def test_c8_short_context_logits_close_to_bf16():
    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    context, num_q = 64, 6
    gk = torch.Generator().manual_seed(11)
    key = torch.randn(context, num_kv_heads, head_dim, generator=gk)
    gv = torch.Generator().manual_seed(12)
    value = torch.randn(context, num_kv_heads, head_dim, generator=gv)
    gq = torch.Generator().manual_seed(13)
    query = torch.randn(num_q, num_q_heads, head_dim, generator=gq)
    gg = torch.Generator().manual_seed(14)
    gate = torch.randn(num_q, num_q_heads, head_dim, generator=gg)
    packed = torch.arange(context).unsqueeze(0).repeat(num_q, 1)
    counts = torch.full((num_q,), context, dtype=torch.int64)

    out_bf16 = qsa_sparse_gqa_attention(query, key, value, gate, packed, counts, num_kv_heads)

    key_c8, _ = qsa_c8.roundtrip_kv_int8(key.reshape(context, num_kv_heads * head_dim), num_kv_heads)
    value_c8, _ = qsa_c8.roundtrip_kv_int8(value.reshape(context, num_kv_heads * head_dim), num_kv_heads)
    key_c8 = key_c8.reshape(context, num_kv_heads, head_dim)
    value_c8 = value_c8.reshape(context, num_kv_heads, head_dim)
    out_c8 = qsa_sparse_gqa_attention(query, key_c8, value_c8, gate, packed, counts, num_kv_heads)

    rel = qsa_c8.logits_relative_error(out_bf16, out_c8)
    assert rel < qsa_c8.C8_LOGITS_REL


# =========================================================================
# C8 main-K/V spec materializes; C8 bytes exactly half BF16 (kv_cache hook)
# =========================================================================
def test_c8_main_kv_spec_is_half_of_bf16():
    bf16 = kvc.make_qsa_main_kv_spec()
    c8 = kvc.make_qsa_main_kv_spec(c8=True)
    assert isinstance(bf16, FullAttentionSpec) and isinstance(c8, FullAttentionSpec)
    assert bf16.dtype == POLICY.qsa_main_dtype
    assert c8.dtype == kvc.QSA_C8_MAIN_DTYPE == torch.int8
    assert get_dtype_size(c8.dtype) == 1 and get_dtype_size(bf16.dtype) == 2
    # Full-attention page stores K + V: 2 * block * heads * dim * elt.
    expected_bf16 = 2 * 128 * kvc.QSA_MAIN_KV_HEADS * kvc.QSA_MAIN_HEAD_DIM * 2
    assert bf16.page_size_bytes == expected_bf16 == 262_144
    assert c8.page_size_bytes == expected_bf16 // 2 == 131_072  # exactly half


def test_c8_main_kv_1m_bytes_match_prd():
    table = kvc.qsa_main_kv_bytes_table()
    bf16 = table["BF16"]
    c8 = table["C8"]
    # Geometry / block accounting.
    assert bf16["max_model_len"] == 1_048_576
    assert bf16["num_blocks"] == 8_192
    assert bf16["num_qsa_layers"] == 12
    assert bf16["shard_ranks"] == 4
    # BF16: 2.00 GiB/layer, 24.00 GiB aggregate, 6.00 GiB/chip (PRD §6 row).
    assert bf16["per_layer_bytes"] == 2 * _GIB
    assert bf16["aggregate_bytes"] == 24 * _GIB
    assert bf16["per_chip_bytes"] == 6 * _GIB
    # C8 halves every figure: 1.00 GiB/layer, 12.00 GiB aggregate, 3.00 GiB/chip.
    assert c8["per_layer_bytes"] == 1 * _GIB
    assert c8["aggregate_bytes"] == 12 * _GIB
    assert c8["per_chip_bytes"] == 3 * _GIB
    assert c8["element_size_bytes"] == 1 and bf16["element_size_bytes"] == 2


def test_c8_main_kv_is_exactly_half_bf16_everywhere():
    table = kvc.qsa_main_kv_bytes_table()
    for field in ("page_bytes", "per_layer_bytes", "aggregate_bytes", "per_chip_bytes"):
        assert table["C8"][field] * 2 == table["BF16"][field]


def test_main_kv_bytes_scale_with_layers_and_shards():
    table = kvc.qsa_main_kv_bytes_table(num_qsa_layers=1, shard_ranks=1)
    assert table["BF16"]["aggregate_bytes"] == table["BF16"]["per_layer_bytes"]
    assert table["BF16"]["per_chip_bytes"] == table["BF16"]["aggregate_bytes"]
    with pytest.raises(ValueError):
        kvc.qsa_main_kv_bytes_per_chip(shard_ranks=0)
