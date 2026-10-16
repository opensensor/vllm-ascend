# SPDX-License-Identifier: Apache-2.0
"""Exact short-prefix optimization, including the boundary to scored selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops import qsa_indexer as indexer


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [3, 18, 65])
@pytest.mark.parametrize("capacity", [32, 512, 544])
@pytest.mark.parametrize("policy", [indexer.QSA_SELECTION_STABLE_ARGSORT, indexer.QSA_SELECTION_FAST_TOPK])
def test_short_prefix_matches_scored_repair_and_over_budget_still_scores(
    monkeypatch, rows, capacity, policy, position_dtype
):
    positions = torch.tensor(([-2, -1, 0, 3, 127, 2047, 2175] * rows)[:rows], dtype=position_dtype)
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn(rows, capacity, generator=generator)
    scores[0].fill_(float("nan"))
    scores[-1].fill_(float("inf"))
    visible, starts, tails = indexer._qsa_position_geometry(positions, 4, capacity)
    selected_width = min(capacity, 512)
    expected = indexer._repair_native_group_indices(
        indexer._select_topk_indices(scores, selected_width, policy).int(), visible
    )
    calls = []

    def score(*args):
        calls.append(args)
        return scores

    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(npu_qsa_indexer_score_310=score))
    tensor = torch.zeros(rows, 2, 16)
    query = SimpleNamespace(device=SimpleNamespace(type="npu"), ndim=3, shape=tensor.shape, contiguous=lambda: tensor)
    result = indexer.qsa_indexer_select_groups_310(
        query,
        torch.full((1, 35, 16), float("nan")),
        torch.zeros(1, capacity // 32, dtype=torch.int32),
        torch.tensor([0, rows], dtype=torch.int32),
        positions,
        compress_ratio=4,
        token_topk=2048,
        selection_policy=policy,
        max_matmul_decode_tokens=0,
    )
    assert len(calls) == (capacity > 512)
    for actual, wanted in zip(
        (result.group_indices, result.group_counts, result.tail_starts, result.tail_counts),
        (expected, visible.clamp_max(selected_width), starts, tails),
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert result.group_indices.is_contiguous()


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_matmul_prefix_preserves_compact_indices_and_geometry(monkeypatch, position_dtype):
    rows, capacity = 128, 32
    positions = torch.arange(rows, dtype=position_dtype)
    query = SimpleNamespace(device=SimpleNamespace(type="npu"), ndim=3, shape=(rows, 2, 16))

    def forbidden(*args):
        pytest.fail("complete-prefix selection must not score")

    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(npu_qsa_indexer_score_310=forbidden))
    monkeypatch.setattr(indexer, "_qsa_matmul_select_groups", forbidden)
    result = indexer.qsa_indexer_select_groups_310(
        query,
        torch.zeros(1, 35, 16),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([0, rows], dtype=torch.int32),
        positions,
        compress_ratio=4,
        token_topk=2048,
    )
    expected = torch.arange(capacity, dtype=torch.int32).expand(rows, -1)
    torch.testing.assert_close(result.group_indices, expected, rtol=0, atol=0)
    assert result.group_counts.dtype == position_dtype
