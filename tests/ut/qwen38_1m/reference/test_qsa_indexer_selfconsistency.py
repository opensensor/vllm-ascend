# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the QSA indexer reference.

Acceptance (T0.6): indexer selection sets stable/deterministic at boundary
lengths (1, ratio-1, ratio, ratio+1, partial group, exactly 2048, 2049+); the
2,048-token budget semantics.
"""

import pytest
import torch

from tests.ut.qwen38_1m.reference.qsa_indexer_reference import (
    INDEXER_BUDGET,
    INDEXER_COMPRESS_RATIO,
    compress_keys,
    expand_selection,
    indexer_scores,
    qsa_select_tokens,
    select_blocks,
    selected_token_set,
    visible_block_count,
)
from tests.ut.qwen38_1m.reference.tolerances import (
    QSA_INDEXER_SCORE_ATOL,
    QSA_INDEXER_SCORE_RTOL,
)

_RATIO = INDEXER_COMPRESS_RATIO  # 4
_DIM = 8
_HEADS = 4


def _rand(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


def test_compression_mean_pooling():
    raw = _rand((13, _DIM), 1)
    compressed = compress_keys(raw, _RATIO)
    assert compressed.shape == (3, _DIM)  # 13 // 4 == 3 complete blocks
    for m in range(3):
        expected = raw[m * _RATIO : (m + 1) * _RATIO].mean(dim=0)
        torch.testing.assert_close(compressed[m], expected)


def test_scores_match_bruteforce():
    query = _rand((5, _HEADS, _DIM), 2)
    keys = _rand((7, _DIM), 3)
    scores = indexer_scores(query, keys)
    expected = torch.zeros(5, 7, dtype=torch.float64)
    for t in range(5):
        for n in range(7):
            acc = 0.0
            for h in range(_HEADS):
                dot = (query[t, h] * keys[n]).sum()
                acc = acc + max(dot.item(), 0.0)
            expected[t, n] = acc
    torch.testing.assert_close(scores, expected, rtol=QSA_INDEXER_SCORE_RTOL, atol=QSA_INDEXER_SCORE_ATOL)


@pytest.mark.parametrize(
    "pos,expected_visible",
    [
        (0, 0),  # length 1: no complete block yet
        (_RATIO - 2, 0),  # ratio-1 tokens: still partial
        (_RATIO - 1, 1),  # exactly ratio tokens: one complete block
        (_RATIO, 1),  # ratio+1: one complete block + partial
        (2 * _RATIO - 1, 2),  # two complete blocks
    ],
)
def test_visible_block_boundaries(pos, expected_visible):
    assert visible_block_count(pos, _RATIO) == expected_visible


def test_selection_is_full_prefix_within_budget():
    """When visible_blocks <= block_topk, the selection is the full causal prefix."""
    token_topk = 16  # block_topk = 4
    seq = 15  # positions 0..14 -> up to 3 complete blocks (< block_topk)
    raw = _rand((seq, _DIM), 5)
    query = _rand((seq, _HEADS, _DIM), 6)
    positions = torch.arange(seq)
    packed, counts = qsa_select_tokens(query, raw, positions, compress_ratio=_RATIO, token_topk=token_topk)
    for t in range(seq):
        selected = selected_token_set(packed[t])
        # Full causal prefix [0, t].
        assert selected == set(range(t + 1))
        assert int(counts[t].item()) == t + 1


def test_budget_prunes_beyond_capacity():
    """When visible_blocks > block_topk, the budget drops the lowest-score blocks."""
    token_topk = 16  # block_topk = 4
    block_topk = token_topk // _RATIO
    seq = 40  # position 39 -> visible = 40 // 4 = 10 blocks > 4
    raw = _rand((seq, _DIM), 7)
    query = _rand((seq, _HEADS, _DIM), 8)
    pos = seq - 1
    positions = torch.tensor([pos])
    packed, counts = qsa_select_tokens(
        query[pos : pos + 1], raw, positions, compress_ratio=_RATIO, token_topk=token_topk
    )
    selected = selected_token_set(packed[0])
    # complete_blocks capped at block_topk -> expanded tokens = block_topk*ratio.
    tail_start = ((pos + 1) // _RATIO) * _RATIO
    tail_count = (pos + 1) - tail_start
    assert int(counts[0].item()) == block_topk * _RATIO + tail_count
    assert len(selected) == block_topk * _RATIO + tail_count
    assert len(selected) < pos + 1  # strictly pruned vs dense

    # The kept blocks are exactly the top-`block_topk` by score.
    compressed = compress_keys(raw, _RATIO)
    scores = indexer_scores(query[pos : pos + 1], compressed)[0]
    top = set(select_blocks(scores, visible_block_count(pos, _RATIO), block_topk))
    kept_blocks = {tok // _RATIO for tok in selected if tok < tail_start}
    assert kept_blocks == top


def test_selection_is_deterministic():
    token_topk = 16
    seq = 40
    raw = _rand((seq, _DIM), 9)
    query = _rand((seq, _HEADS, _DIM), 10)
    positions = torch.arange(seq)
    first = qsa_select_tokens(query, raw, positions, compress_ratio=_RATIO, token_topk=token_topk)
    second = qsa_select_tokens(query, raw, positions, compress_ratio=_RATIO, token_topk=token_topk)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.parametrize("pos", [INDEXER_BUDGET - 1, INDEXER_BUDGET, INDEXER_BUDGET + 3])
def test_budget_2048_boundary(pos):
    """Boundary at exactly the 2048 budget and just beyond it."""
    ratio = _RATIO
    block_topk = INDEXER_BUDGET // ratio  # 512
    seq = pos + 1
    # Tiny head geometry keeps the 2048-scale test cheap.
    raw = _rand((seq, 4), seed=1000 + pos)
    query = _rand((1, 1, 4), seed=2000 + pos)
    positions = torch.tensor([pos])
    packed, counts = qsa_select_tokens(query, raw, positions, compress_ratio=ratio, token_topk=INDEXER_BUDGET)
    vis = visible_block_count(pos, ratio)
    tail_start = ((pos + 1) // ratio) * ratio
    tail_count = (pos + 1) - tail_start
    complete = min(vis, block_topk)
    assert int(counts[0].item()) == complete * ratio + tail_count
    if vis <= block_topk:
        # Within budget -> dense causal prefix.
        assert selected_token_set(packed[0]) == set(range(pos + 1))
    else:
        # Over budget -> pruned.
        assert len(selected_token_set(packed[0])) == complete * ratio + tail_count
        assert len(selected_token_set(packed[0])) < pos + 1


def test_expand_pads_and_bounds():
    """Packed width and padding match the expand-kernel contract."""
    token_topk = 16
    selected = [3, 1, 0]  # fewer than block_topk (4)
    pos = 14
    packed, count = expand_selection(selected, pos, _RATIO, token_topk)
    assert packed.numel() == token_topk + _RATIO - 1
    # 3 blocks * 4 + tail. pos+1=15, tail_start=12, tail_count=3.
    assert count == 3 * _RATIO + 3
    # trailing columns are padded with -1
    assert int(packed[-1].item()) == -1
