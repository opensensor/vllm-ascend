# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity tests for the Ascend 310P Qwen4Exp QSA indexer (plan T6.1).

The device-free torch indexer (:class:`AscendQwen4ExpQSAIndexer`) must produce
selection sets *identical* to the T0.6 eager brute-force reference
(``tests/ut/qwen38_1m/reference/qsa_indexer_reference.py``) at every boundary
length, for repeated-block and final-token cases, and with a bitwise-stable
deterministic top-k.

The indexer stores q / k and its side caches in the policy's float16 dtype and
accumulates scores in float32. Score-dependent (over-budget) cases below use
small-integer key ramps that are exact in float16, so the parity isolates the
*selection logic* rather than float precision; within-budget cases are dense and
precision-independent by construction.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_qsa_indexer.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.reference.qsa_indexer_reference import (
    INDEXER_BUDGET,
    INDEXER_COMPRESS_RATIO,
    qsa_select_tokens,
    selected_token_set,
)
from vllm_ascend.models.qwen4_exp.indexer_qsa import (
    AscendQwen4ExpQSAIndexer,
    QSAIndexerOutput,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_cache import (
    PAD_SLOT_ID,
    build_qsa_indexer_metadata,
    compressed_qsa_slot_mapping,
    qsa_gather_rows,
    qsa_scatter_rows,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    _stable_topk_indices,
    expand_group_selection,
    qsa_indexer_score_310_reference,
    qsa_indexer_select_groups,
)

_RATIO = INDEXER_COMPRESS_RATIO  # 4


def _make_indexer(*, budget: int, ratio: int = _RATIO) -> AscendQwen4ExpQSAIndexer:
    config = SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        num_speculative_tokens=0,
    )
    return AscendQwen4ExpQSAIndexer(config=config, layer_idx=0)


def _rand(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


def _module_sets(out: QSAIndexerOutput) -> list[set[int]]:
    return [selected_token_set(row) for row in out.token_indices]


def _reference_sets(query, raw, positions, *, ratio, budget) -> list[set[int]]:
    packed, _ = qsa_select_tokens(query, raw, positions, compress_ratio=ratio, token_topk=budget)
    return [selected_token_set(row) for row in packed]


def _assert_parity(indexer, query, raw, positions):
    ratio = indexer.compress_ratio
    budget = indexer.token_topk
    out = indexer.forward(query, raw, positions)
    ref_packed, ref_counts = qsa_select_tokens(query, raw, positions, compress_ratio=ratio, token_topk=budget)
    for t in range(query.shape[0]):
        assert selected_token_set(out.token_indices[t]) == selected_token_set(ref_packed[t]), (
            f"selection set mismatch at query {t}"
        )
        assert int(out.valid_counts[t].item()) == int(ref_counts[t].item())
    return out


# ---------------------------------------------------------------------------
# Boundary lengths (final-token query): 1, ratio-1, ratio, ratio+1, partial.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seq_len",
    [1, _RATIO - 1, _RATIO, _RATIO + 1, _RATIO + 2, 3 * _RATIO + 2],
)
def test_boundary_lengths_final_token(seq_len):
    """Final-token selection matches the reference at short boundary lengths."""
    indexer = _make_indexer(budget=16, ratio=_RATIO)
    raw = _rand((seq_len, 8), seed=100 + seq_len)
    query = _rand((1, 4, 8), seed=200 + seq_len)
    positions = torch.tensor([seq_len - 1])
    _assert_parity(indexer, query, raw, positions)


def test_all_query_tokens_within_budget_is_dense():
    """Every token within budget selects its full causal prefix (precision-free)."""
    indexer = _make_indexer(budget=16, ratio=_RATIO)  # block_topk = 4
    seq = 15  # up to 3 complete blocks < block_topk -> always dense
    raw = _rand((seq, 8), seed=5)
    query = _rand((seq, 4, 8), seed=6)
    positions = torch.arange(seq)
    out = _assert_parity(indexer, query, raw, positions)
    for t in range(seq):
        assert selected_token_set(out.token_indices[t]) == set(range(t + 1))
        assert int(out.valid_counts[t].item()) == t + 1


# ---------------------------------------------------------------------------
# Over-budget pruning: distinct integer key ramp (exact in float16).
# ---------------------------------------------------------------------------
def _ramp_inputs(num_blocks, ratio, tail, *, dim=4):
    """Keys whose block ``m`` has value ``m+1`` on dim 0; query = e0 (1 head).

    Scores are ``score[m] = m + 1`` -- distinct small integers exact in float16,
    so the top-k is unambiguous and identical in float64 and float16.
    """
    seq_len = num_blocks * ratio + tail
    raw = torch.zeros((seq_len, dim), dtype=torch.float64)
    for m in range(num_blocks):
        raw[m * ratio : (m + 1) * ratio, 0] = float(m + 1)
    # Partial-tail tokens (not compressed) get a marker on another dim.
    for s in range(num_blocks * ratio, seq_len):
        raw[s, 1] = 1.0
    query = torch.zeros((1, 1, dim), dtype=torch.float64)
    query[0, 0, 0] = 1.0
    return raw, query, seq_len


def test_over_budget_prunes_lowest_scoring_blocks():
    indexer = _make_indexer(budget=16, ratio=_RATIO)  # block_topk = 4
    num_blocks = 10  # > block_topk (4) -> pruned
    raw, query, seq_len = _ramp_inputs(num_blocks, _RATIO, tail=2)
    positions = torch.tensor([seq_len - 1])
    out = _assert_parity(indexer, query, raw, positions)
    selected = selected_token_set(out.token_indices[0])
    tail_start = (seq_len // _RATIO) * _RATIO
    # Highest-scoring 4 blocks are 6,7,8,9 -> tokens 24..39, plus the 2-token tail.
    kept_blocks = {tok // _RATIO for tok in selected if tok < tail_start}
    assert kept_blocks == {6, 7, 8, 9}
    assert len(selected) == 4 * _RATIO + (seq_len - tail_start)


@pytest.mark.parametrize("seq_len", [INDEXER_BUDGET, INDEXER_BUDGET + 6])
def test_budget_2048_boundary(seq_len):
    """Exactly-2048 (dense) and 2049+ (pruned) at the real 2,048 budget."""
    indexer = _make_indexer(budget=INDEXER_BUDGET, ratio=_RATIO)
    num_blocks = seq_len // _RATIO
    tail = seq_len - num_blocks * _RATIO
    raw, query, _ = _ramp_inputs(num_blocks, _RATIO, tail=tail)
    positions = torch.tensor([seq_len - 1])
    out = _assert_parity(indexer, query, raw, positions)
    block_topk = INDEXER_BUDGET // _RATIO
    if num_blocks <= block_topk:
        assert selected_token_set(out.token_indices[0]) == set(range(seq_len))
    else:
        selected = selected_token_set(out.token_indices[0])
        assert len(selected) < seq_len  # strictly pruned


# ---------------------------------------------------------------------------
# Repeated-block ties: deterministic ascending-index tie-break.
# ---------------------------------------------------------------------------
def test_repeated_block_tie_break_matches_reference():
    """All-identical blocks tie; both keep the lowest-index blocks."""
    indexer = _make_indexer(budget=16, ratio=_RATIO)  # block_topk = 4
    num_blocks = 9  # > block_topk
    seq_len = num_blocks * _RATIO
    raw = torch.ones((seq_len, 4), dtype=torch.float64)  # every block identical
    query = torch.ones((1, 2, 4), dtype=torch.float64)
    positions = torch.tensor([seq_len - 1])
    out = _assert_parity(indexer, query, raw, positions)
    kept_blocks = {tok // _RATIO for tok in selected_token_set(out.token_indices[0])}
    assert kept_blocks == {0, 1, 2, 3}  # ties -> lowest indices


def test_partial_repeated_block_ties():
    """Two score tiers with intra-tier ties resolve by ascending index."""
    indexer = _make_indexer(budget=16, ratio=_RATIO)  # block_topk = 4
    num_blocks = 8
    seq_len = num_blocks * _RATIO
    raw = torch.zeros((seq_len, 4), dtype=torch.float64)
    # Blocks 0..3 score 2, blocks 4..7 score 1: high tier fully selected.
    for m in range(num_blocks):
        raw[m * _RATIO : (m + 1) * _RATIO, 0] = 2.0 if m < 4 else 1.0
    query = torch.zeros((1, 1, 4), dtype=torch.float64)
    query[0, 0, 0] = 1.0
    positions = torch.tensor([seq_len - 1])
    out = _assert_parity(indexer, query, raw, positions)
    kept_blocks = {tok // _RATIO for tok in selected_token_set(out.token_indices[0])}
    assert kept_blocks == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "scores,k",
    [
        ([[1.0, 1.0, 1.0, 1.0, 1.0]], 3),
        ([[4.0, 3.0, 3.0, 3.0, 2.0, 1.0]], 3),
        ([[2.0, -torch.inf, -torch.inf, -torch.inf]], 3),
        ([[0.0, 2.0, 1.0, 2.0], [5.0, 4.0, 3.0, 2.0]], 2),
    ],
)
def test_bounded_stable_topk_matches_full_stable_sort(scores, k):
    score_tensor = torch.tensor(scores)
    expected = torch.argsort(score_tensor, dim=1, descending=True, stable=True)[:, :k]
    assert torch.equal(_stable_topk_indices(score_tensor, k), expected)


# ---------------------------------------------------------------------------
# Determinism: bitwise-stable across two runs.
# ---------------------------------------------------------------------------
def test_topk_is_bitwise_deterministic():
    indexer = _make_indexer(budget=16, ratio=_RATIO)
    seq = 40
    raw = _rand((seq, 8), seed=9)
    query = _rand((seq, 4, 8), seed=10)
    positions = torch.arange(seq)
    first = indexer.forward(query, raw, positions)
    second = indexer.forward(query, raw, positions)
    assert torch.equal(first.token_indices, second.token_indices)
    assert torch.equal(first.valid_counts, second.valid_counts)
    assert torch.equal(first.packed, second.packed)


def test_compact_group_selection_expands_to_reference_tokens():
    """The native-kernel contract retains exact token-level QSA semantics."""
    ratio = 4
    budget = 16
    raw, query, seq_len = _ramp_inputs(num_blocks=10, ratio=ratio, tail=2)
    compressed = raw[: (seq_len // ratio) * ratio].view(-1, ratio, raw.shape[-1]).mean(dim=1)
    positions = torch.tensor([seq_len - 1])

    selection = qsa_indexer_select_groups(
        query,
        compressed,
        positions,
        compress_ratio=ratio,
        token_topk=budget,
        accum_dtype=torch.float64,
    )
    packed, counts = expand_group_selection(selection, ratio, budget)
    reference, reference_counts = qsa_select_tokens(
        query,
        raw,
        positions,
        compress_ratio=ratio,
        token_topk=budget,
    )

    assert selection.group_indices.shape == (1, budget // ratio)
    assert selection.group_counts.tolist() == [budget // ratio]
    assert selection.tail_starts.tolist() == [40]
    assert selection.tail_counts.tolist() == [2]
    assert torch.equal(packed, reference)
    assert torch.equal(counts, reference_counts)


# ---------------------------------------------------------------------------
# Packed buffer contract (T6.2-facing): trailing count column.
# ---------------------------------------------------------------------------
def test_packed_buffer_has_trailing_count_column():
    indexer = _make_indexer(budget=16, ratio=_RATIO)
    seq = 10
    raw = _rand((seq, 8), seed=11)
    query = _rand((seq, 4, 8), seed=12)
    positions = torch.arange(seq)
    out = indexer.forward(query, raw, positions)
    packed = out.packed
    assert packed.shape == (seq, indexer.packed_output_width)
    assert packed.dtype == torch.int32
    for t in range(seq):
        assert int(packed[t, -1].item()) == int(out.valid_counts[t].item())
        assert torch.equal(packed[t, : indexer.output_width], out.token_indices[t].to(torch.int32))


# ---------------------------------------------------------------------------
# Side-cache slot mappings / metadata torch fallback (Triton-free port).
# ---------------------------------------------------------------------------
def test_metadata_torch_fallback_two_request_batch():
    ratio = 2
    ring_size = 2
    storage_block_size = 8
    query_start_loc = torch.tensor([0, 6, 11], dtype=torch.int32)
    seq_lens = torch.tensor([6, 5], dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    num_tokens = 11
    meta = build_qsa_indexer_metadata(
        query_start_loc,
        seq_lens,
        block_table,
        num_tokens,
        compress_ratio=ratio,
        ring_size=ring_size,
        storage_block_size=storage_block_size,
    )
    assert meta.token_to_req.tolist() == [0] * 6 + [1] * 5
    assert meta.logical_positions.tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    # visible = min((pos+1)//ratio, seq_len//ratio)
    assert meta.visible_blocks.tolist() == [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2]
    # Compressed rows only at group-closing (odd) positions.
    p = PAD_SLOT_ID
    assert meta.compressed_slot_mapping.tolist() == [p, 0, p, 1, p, 2, p, 8, p, 9, p]
    # Ring keeps only the trailing ring_size tokens of each request.
    assert meta.ring_slot_mapping.tolist() == [p, p, p, p, 0, 1, p, p, p, 3, 2]


def test_compressed_slot_mapping_boundary_rule():
    ratio = 4
    logical_positions = torch.arange(10)
    token_to_req = torch.zeros(10, dtype=torch.long)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    slots = compressed_qsa_slot_mapping(block_table, token_to_req, logical_positions, 8, ratio)
    # Boundaries at pos 3 (->0) and 7 (->1); all others PAD.
    expected = [PAD_SLOT_ID] * 10
    expected[3] = 0
    expected[7] = 1
    assert slots.tolist() == expected


def test_scatter_gather_round_trip_skips_pad():
    cache = torch.zeros((4, 3))
    rows = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float32)
    slots = torch.tensor([2, PAD_SLOT_ID, 0], dtype=torch.long)
    qsa_scatter_rows(cache, slots, rows)
    assert torch.equal(cache[2], rows[0])
    assert torch.equal(cache[0], rows[2])
    assert torch.equal(cache[1], torch.zeros(3))  # untouched by the PAD row
    gathered = qsa_gather_rows(cache, torch.tensor([0, PAD_SLOT_ID, 2], dtype=torch.long))
    assert torch.equal(gathered[0], rows[2])
    assert torch.equal(gathered[1], torch.zeros(3))  # PAD -> zeros
    assert torch.equal(gathered[2], rows[0])


def test_ring_cache_write_retains_open_group_suffix():
    """The raw-key ring holds the trailing ``ring_size`` keys of the context."""
    indexer = _make_indexer(budget=16, ratio=_RATIO)  # ring_size == ratio == 4
    seq = 10
    raw = _rand((seq, 8), seed=21)
    query = _rand((1, 4, 8), seed=22)
    positions = torch.tensor([seq - 1])
    out = indexer.forward(query, raw, positions)
    assert out.ring_cache.shape == (indexer.ring_size, 8)
    # Last ring_size tokens land at slot (pos % ring_size).
    for pos in range(seq - indexer.ring_size, seq):
        slot = pos % indexer.ring_size
        expected = raw[pos].to(out.ring_cache.dtype)
        torch.testing.assert_close(out.ring_cache[slot], expected)


def test_single_kv_head_enforced():
    indexer = _make_indexer(budget=16, ratio=_RATIO)
    raw = torch.zeros((8, 2, 8))  # two kv heads -> rejected
    query = torch.zeros((1, 4, 8))
    with pytest.raises(ValueError):
        indexer.forward(query, raw, torch.tensor([7]))


def test_310_score_reference_uses_paged_groups_and_ignores_scratch():
    groups_per_block = 2
    scratch_rows = 3
    cache = torch.zeros((3, groups_per_block + scratch_rows, 2), dtype=torch.float16)
    cache[2, 0] = torch.tensor([1, 0])
    cache[2, 1] = torch.tensor([0, 2])
    cache[0, 0] = torch.tensor([3, 0])
    cache[0, 1] = torch.tensor([0, 4])
    cache[:, groups_per_block:] = 10_000  # must never enter the score address space
    query = torch.tensor([[[1, 1], [-1, 1]]], dtype=torch.float16)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)

    scores = qsa_indexer_score_310_reference(
        query,
        cache,
        block_table,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([15], dtype=torch.int32),
        compress_ratio=4,
    )

    # Logical groups follow the permuted physical pages: [2:0, 2:1, 0:0, 0:1].
    torch.testing.assert_close(scores, torch.tensor([[1, 4, 3, 8]], dtype=torch.float32))
