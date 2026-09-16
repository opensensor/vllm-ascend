# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the indexer / CSA2 reference (E0.4, priority 4).

Acceptance: indexer selection vs. brute force at boundary lengths; scoring and
compression self-consistent.
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.indexer_reference import (
    INDEXER_HEAD_DIM,
    INDEXER_N_HEADS,
    INDEXER_TOPK,
    bruteforce_compress_keys,
    bruteforce_indexer_scores,
    bruteforce_select_blocks,
    compress_keys,
    indexer_scores,
    indexer_select,
    select_blocks,
    visible_block_count,
)
from tests.ut.deepseek_w2.reference.tolerances import (
    INDEXER_COMPRESS_ATOL,
    INDEXER_COMPRESS_RTOL,
    INDEXER_SCORE_ATOL,
    INDEXER_SCORE_RTOL,
)

_SCALE = INDEXER_HEAD_DIM**-0.5


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


# --- CSA2 compression -------------------------------------------------------


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("seq_len", [4, 7, 16, 17])
def test_compress_matches_bruteforce(compress_ratio, seq_len):
    raw = _rand((seq_len, INDEXER_HEAD_DIM), seed=seq_len + compress_ratio)
    vec = compress_keys(raw, compress_ratio)
    brute = bruteforce_compress_keys(raw, compress_ratio)
    assert vec.shape[0] == seq_len // compress_ratio
    torch.testing.assert_close(vec, brute, rtol=INDEXER_COMPRESS_RTOL, atol=INDEXER_COMPRESS_ATOL)


def test_ratio_one_is_identity():
    raw = _rand((5, INDEXER_HEAD_DIM), seed=1)
    torch.testing.assert_close(compress_keys(raw, 1), raw, rtol=0, atol=INDEXER_COMPRESS_ATOL)


# --- Lightning-Indexer scoring ---------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_match_bruteforce(seed):
    num_blocks = 6
    query = _rand((5, INDEXER_N_HEADS, INDEXER_HEAD_DIM), seed)
    weights = _rand((5, INDEXER_N_HEADS), seed + 1).abs()
    keys = _rand((num_blocks, INDEXER_HEAD_DIM), seed + 2)
    vec = indexer_scores(query, weights, keys, _SCALE)
    brute = bruteforce_indexer_scores(query, weights, keys, _SCALE)
    torch.testing.assert_close(vec, brute, rtol=INDEXER_SCORE_RTOL, atol=INDEXER_SCORE_ATOL)


def test_scores_are_nonnegative_with_nonneg_weights():
    query = _rand((4, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 10)
    weights = _rand((4, INDEXER_N_HEADS), 11).abs()
    keys = _rand((5, INDEXER_HEAD_DIM), 12)
    scores = indexer_scores(query, weights, keys, _SCALE)
    assert torch.all(scores >= 0.0)


def test_relu_gates_negative_dots():
    """A key anti-aligned with every head contributes zero (relu gate)."""
    query = torch.ones(1, 2, 4, dtype=torch.float64)
    weights = torch.ones(1, 2, dtype=torch.float64)
    keys = torch.stack([torch.ones(4, dtype=torch.float64), -torch.ones(4, dtype=torch.float64)])
    scores = indexer_scores(query, weights, keys, 1.0)
    assert scores[0, 0] > 0.0
    assert scores[0, 1] == 0.0


# --- Selection at boundary lengths -----------------------------------------


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_selection_matches_bruteforce(compress_ratio):
    seq_len = 20
    query = _rand((seq_len, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 3)
    weights = _rand((seq_len, INDEXER_N_HEADS), 4).abs()
    raw = _rand((seq_len, INDEXER_HEAD_DIM), 5)
    compressed = compress_keys(raw, compress_ratio)
    scores = indexer_scores(query, weights, compressed, _SCALE)
    block_topk = INDEXER_TOPK // compress_ratio
    for t in range(seq_len):
        vis = min(visible_block_count(t, compress_ratio), scores.shape[1])
        got = set(select_blocks(scores[t], vis, block_topk))
        expected = bruteforce_select_blocks(scores[t], vis, block_topk)
        assert got == expected


def test_short_context_selects_all_visible():
    """Boundary: fewer visible blocks than the budget -> dense (all kept)."""
    compress_ratio = 2
    block_topk = INDEXER_TOPK // compress_ratio
    # Position 5 -> visible blocks = 3 < block_topk (4): all selected.
    pos = 5
    vis = visible_block_count(pos, compress_ratio)
    assert vis < block_topk
    scores = _rand((vis,), seed=7).squeeze()
    got = select_blocks(scores, vis, block_topk)
    assert set(got) == set(range(vis))


def test_exact_budget_boundary():
    """Boundary: visible blocks exactly equal to the budget."""
    block_topk = INDEXER_TOPK  # 8
    vis = block_topk
    scores = _rand((vis,), seed=8).squeeze()
    got = select_blocks(scores, vis, block_topk)
    assert set(got) == set(range(vis))


def test_over_budget_prunes_to_topk():
    """Boundary: more visible blocks than budget -> exactly block_topk kept."""
    block_topk = INDEXER_TOPK
    vis = block_topk + 5
    scores = torch.arange(vis, dtype=torch.float64)  # strictly increasing
    got = select_blocks(scores, vis, block_topk)
    assert len(got) == block_topk
    # Highest-scoring blocks are the last indices.
    assert set(got) == set(range(vis - block_topk, vis))


def test_full_path_causal_visibility():
    """indexer_select never selects a block beyond the causal visible range."""
    seq_len = 16
    compress_ratio = 2
    query = _rand((seq_len, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 20)
    weights = _rand((seq_len, INDEXER_N_HEADS), 21).abs()
    raw = _rand((seq_len, INDEXER_HEAD_DIM), 22)
    positions = torch.arange(seq_len)
    selected = indexer_select(query, raw, positions, weights, compress_ratio=compress_ratio)
    for t in range(seq_len):
        vis = visible_block_count(t, compress_ratio)
        for block in selected[t]:
            assert block < vis
