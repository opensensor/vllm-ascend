# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Ascend 310P DeepSeek V4.1 sparse-attention indexer (E3.2).

Validates the host-eager adaptation of the shipped V4 indexer against the E0.4
pure-PyTorch reference (``indexer_reference``): mean-pool CSA2 compression,
Lightning-Indexer scoring, and deterministic top-k selection at boundary
lengths, plus the ratio-{0,1,2} gate that is the V4.1 adaptation of the shipped
``compress_ratio == 4`` instantiation gate.
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.indexer_reference import (
    INDEXER_HEAD_DIM,
    INDEXER_N_HEADS,
    INDEXER_TOPK,
    bruteforce_select_blocks,
    compress_keys,
    indexer_scores,
    indexer_select,
    visible_block_count,
)
from tests.ut.deepseek_w2.reference.tolerances import (
    INDEXER_COMPRESS_ATOL,
    INDEXER_COMPRESS_RTOL,
    INDEXER_SCORE_ATOL,
    INDEXER_SCORE_RTOL,
)
from vllm_ascend.models.deepseek_v41.dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
)
from vllm_ascend.models.deepseek_v41.indexer import (
    ACTIVE_COMPRESS_RATIOS,
    SELECTION_PAD,
    SUPPORTED_COMPRESS_RATIOS,
    AscendDeepseekV41Indexer,
    block_topk_for_ratio,
    deterministic_topk_blocks,
    lightning_indexer_scores,
    make_indexer_ops,
    mean_pool_compress,
    select_topk_blocks,
)
from vllm_ascend.models.deepseek_v41.indexer import (
    visible_block_count as v41_visible_block_count,
)

_SCALE = INDEXER_HEAD_DIM**-0.5


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


class _Config:
    """Duck-typed stand-in for the HF DeepSeek V4.1 index config fields."""

    def __init__(self, n_heads=INDEXER_N_HEADS, head_dim=INDEXER_HEAD_DIM, index_topk=INDEXER_TOPK,
                 q_lora_rank=None, hidden_size=None):
        self.index_n_heads = n_heads
        self.index_head_dim = head_dim
        self.index_topk = index_topk
        if q_lora_rank is not None:
            self.q_lora_rank = q_lora_rank
        if hidden_size is not None:
            self.hidden_size = hidden_size


def _make_indexer(compress_ratio, **cfg):
    return AscendDeepseekV41Indexer(_Config(**cfg), compress_ratio=compress_ratio, build_projections=False)


# ---------------------------------------------------------------------------
# CSA2 compression parity vs. E0.4 reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("seq_len", [4, 7, 16, 17])
def test_compress_matches_reference(compress_ratio, seq_len):
    raw = _rand((seq_len, INDEXER_HEAD_DIM), seed=seq_len + compress_ratio)
    got = mean_pool_compress(raw, compress_ratio)
    ref = compress_keys(raw, compress_ratio)
    assert got.shape[0] == seq_len // compress_ratio
    torch.testing.assert_close(got, ref, rtol=INDEXER_COMPRESS_RTOL, atol=INDEXER_COMPRESS_ATOL)


def test_ratio_one_compress_is_identity():
    raw = _rand((5, INDEXER_HEAD_DIM), seed=1)
    torch.testing.assert_close(mean_pool_compress(raw, 1), raw, rtol=0, atol=INDEXER_COMPRESS_ATOL)


# ---------------------------------------------------------------------------
# Lightning-Indexer scoring parity vs. E0.4 reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_match_reference(seed):
    query = _rand((5, INDEXER_N_HEADS, INDEXER_HEAD_DIM), seed)
    weights = _rand((5, INDEXER_N_HEADS), seed + 1).abs()
    keys = _rand((6, INDEXER_HEAD_DIM), seed + 2)
    got = lightning_indexer_scores(query, weights, keys, _SCALE)
    ref = indexer_scores(query, weights, keys, _SCALE)
    torch.testing.assert_close(got, ref, rtol=INDEXER_SCORE_RTOL, atol=INDEXER_SCORE_ATOL)


def test_relu_gates_negative_dots():
    query = torch.ones(1, 2, 4, dtype=torch.float64)
    weights = torch.ones(1, 2, dtype=torch.float64)
    keys = torch.stack([torch.ones(4, dtype=torch.float64), -torch.ones(4, dtype=torch.float64)])
    scores = lightning_indexer_scores(query, weights, keys, 1.0)
    assert scores[0, 0] > 0.0
    assert scores[0, 1] == 0.0


# ---------------------------------------------------------------------------
# Deterministic top-k selection at boundary lengths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_selection_matches_reference_bruteforce(compress_ratio):
    seq_len = 20
    query = _rand((seq_len, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 3)
    weights = _rand((seq_len, INDEXER_N_HEADS), 4).abs()
    raw = _rand((seq_len, INDEXER_HEAD_DIM), 5)
    compressed = mean_pool_compress(raw, compress_ratio)
    scores = lightning_indexer_scores(query, weights, compressed, _SCALE)
    positions = torch.arange(seq_len)
    got = select_topk_blocks(scores, positions, compress_ratio, block_topk_for_ratio(INDEXER_TOPK, compress_ratio))
    block_topk = block_topk_for_ratio(INDEXER_TOPK, compress_ratio)
    for t in range(seq_len):
        vis = min(v41_visible_block_count(t, compress_ratio), scores.shape[1])
        expected = bruteforce_select_blocks(scores[t], vis, block_topk)
        assert set(got[t]) == expected


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_full_pipeline_matches_reference_indexer_select(compress_ratio):
    """End-to-end module output == E0.4 indexer_select (set equality per token)."""
    seq_len = 24
    query = _rand((seq_len, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 30)
    weights = _rand((seq_len, INDEXER_N_HEADS), 31).abs()
    raw = _rand((seq_len, INDEXER_HEAD_DIM), 32)
    positions = torch.arange(seq_len)

    indexer = _make_indexer(compress_ratio)
    result = indexer(
        hidden_states=None,
        qr=None,
        raw_keys=raw,
        positions=positions,
        precomputed_query=query,
        precomputed_weights=weights,
    )
    ref = indexer_select(query, raw, positions, weights, compress_ratio=compress_ratio, index_topk=INDEXER_TOPK)
    assert len(result.blocks_per_token) == seq_len
    for t in range(seq_len):
        assert set(result.blocks_per_token[t]) == set(ref[t])
        # Descending-score order preserved exactly (not just set equality).
        assert result.blocks_per_token[t] == ref[t]


def test_selection_is_deterministic_on_ties():
    """Equal scores -> ascending block-id tie-break, stable across runs."""
    scores = torch.ones(9, dtype=torch.float64)
    a = deterministic_topk_blocks(scores, visible_blocks=9, block_topk=4)
    b = deterministic_topk_blocks(scores, visible_blocks=9, block_topk=4)
    assert a.tolist() == b.tolist() == [0, 1, 2, 3]


def test_short_context_selects_all_visible():
    compress_ratio = 2
    block_topk = block_topk_for_ratio(INDEXER_TOPK, compress_ratio)
    pos = 5  # visible blocks = 3 < block_topk (4)
    vis = v41_visible_block_count(pos, compress_ratio)
    assert vis < block_topk
    scores = _rand((vis,), seed=7).squeeze(0) if vis == 1 else _rand((vis,), seed=7)
    got = deterministic_topk_blocks(scores, vis, block_topk)
    assert set(got.tolist()) == set(range(vis))


def test_over_budget_prunes_to_topk():
    block_topk = INDEXER_TOPK
    vis = block_topk + 5
    scores = torch.arange(vis, dtype=torch.float64)  # strictly increasing
    got = deterministic_topk_blocks(scores, vis, block_topk)
    assert len(got) == block_topk
    assert set(got.tolist()) == set(range(vis - block_topk, vis))
    # Descending-score order: highest (last index) first.
    assert got[0].item() == vis - 1


def test_full_path_respects_causal_visibility():
    seq_len = 16
    compress_ratio = 2
    query = _rand((seq_len, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 20)
    weights = _rand((seq_len, INDEXER_N_HEADS), 21).abs()
    raw = _rand((seq_len, INDEXER_HEAD_DIM), 22)
    positions = torch.arange(seq_len)
    result = _make_indexer(compress_ratio)(
        hidden_states=None, qr=None, raw_keys=raw, positions=positions,
        precomputed_query=query, precomputed_weights=weights,
    )
    for t in range(seq_len):
        vis = v41_visible_block_count(t, compress_ratio)
        for block in result.blocks_per_token[t]:
            assert block < vis


# ---------------------------------------------------------------------------
# Ratio {0,1,2} gate -- the V4.1 adaptation of the shipped ratio==4 gate
# ---------------------------------------------------------------------------
def test_ratio_zero_is_sliding_window_disabled():
    indexer = _make_indexer(0)
    assert indexer.enabled is False
    assert indexer.block_topk == 0
    out = indexer(hidden_states=None, qr=None, raw_keys=_rand((4, INDEXER_HEAD_DIM), 1),
                  positions=torch.arange(4), precomputed_query=_rand((4, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 2),
                  precomputed_weights=_rand((4, INDEXER_N_HEADS), 3).abs())
    assert out is None


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_active_ratios_enabled(compress_ratio):
    indexer = _make_indexer(compress_ratio)
    assert indexer.enabled is True
    assert AscendDeepseekV41Indexer.supports_ratio(compress_ratio) is True
    assert indexer.block_topk == INDEXER_TOPK // compress_ratio


def test_unsupported_ratio_rejected():
    for bad in (3, 4, 128):
        assert bad not in SUPPORTED_COMPRESS_RATIOS
        with pytest.raises(ValueError):
            _make_indexer(bad)


def test_supports_ratio_matches_active_set():
    for r in ACTIVE_COMPRESS_RATIOS:
        assert AscendDeepseekV41Indexer.supports_ratio(r)
    assert not AscendDeepseekV41Indexer.supports_ratio(0)


# ---------------------------------------------------------------------------
# Dense padded output form + dtype policy wiring
# ---------------------------------------------------------------------------
def test_block_indices_padding_shape():
    compress_ratio = 2
    seq_len = 12
    result = _make_indexer(compress_ratio)(
        hidden_states=None, qr=None, raw_keys=_rand((seq_len, INDEXER_HEAD_DIM), 40),
        positions=torch.arange(seq_len),
        precomputed_query=_rand((seq_len, INDEXER_N_HEADS, INDEXER_HEAD_DIM), 41),
        precomputed_weights=_rand((seq_len, INDEXER_N_HEADS), 42).abs(),
    )
    block_topk = INDEXER_TOPK // compress_ratio
    assert result.block_indices.shape == (seq_len, block_topk)
    assert result.block_indices.dtype == torch.long
    for t in range(seq_len):
        row = result.block_indices[t]
        valid = row[row != SELECTION_PAD].tolist()
        assert valid == result.blocks_per_token[t]


def test_dtype_policy_wiring():
    indexer = _make_indexer(1)
    assert indexer.dtype == ASCEND_DEEPSEEKV41_DTYPE_POLICY.indexer_dtype == torch.float16
    assert indexer.device_accumulation_dtype == ASCEND_DEEPSEEKV41_DTYPE_POLICY.indexer_accumulation_dtype
    assert indexer.accumulation_dtype == torch.float64  # host parity upcast


def test_projections_build_and_run():
    cfg = dict(q_lora_rank=64, hidden_size=96)
    indexer = AscendDeepseekV41Indexer(_Config(**cfg), compress_ratio=2, build_projections=True)
    assert indexer.wq_b is not None and indexer.weights_proj is not None
    assert indexer.wq_b.weight.dtype == torch.float16
    seq_len = 8
    qr = torch.randn(seq_len, 64, dtype=torch.float16)
    hidden = torch.randn(seq_len, 96, dtype=torch.float16)
    raw = torch.randn(seq_len, INDEXER_HEAD_DIM, dtype=torch.float16)
    positions = torch.arange(seq_len)
    result = indexer(hidden_states=hidden, qr=qr, raw_keys=raw, positions=positions)
    assert result is not None
    for t in range(seq_len):
        vis = v41_visible_block_count(t, 2)
        assert len(result.blocks_per_token[t]) == min(vis, indexer.block_topk)


def test_make_indexer_ops_returns_torch_fallback_on_host():
    ops = make_indexer_ops(INDEXER_TOPK, _SCALE, torch.float64)
    # No torch_npu / _C_ascend kernel on the 310P host: torch fallback is used.
    raw = _rand((6, INDEXER_HEAD_DIM), 50)
    compressed = ops.compress_keys(raw, 2)
    torch.testing.assert_close(compressed, compress_keys(raw, 2),
                               rtol=INDEXER_COMPRESS_RTOL, atol=INDEXER_COMPRESS_ATOL)


def test_v41_visible_block_count_matches_reference():
    for pos in range(20):
        for ratio in (1, 2):
            assert v41_visible_block_count(pos, ratio) == visible_block_count(pos, ratio)
