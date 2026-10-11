# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for long cached-prefix QSA selection workspace growth."""

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops import qsa_indexer as indexer


@pytest.mark.parametrize("policy", sorted(indexer.QSA_SELECTION_POLICIES))
@pytest.mark.parametrize("rows", [1, 7, 8, 17])
def test_selection_bounds_backend_rows_and_preserves_order(monkeypatch, policy, rows):
    width, k = 37, 9
    scores = torch.randn(rows, width, generator=torch.Generator().manual_seed(7))
    # Include full ties and masked groups. The fast policy retains its own
    # cutoff tie behavior; the stable policy must retain lower-index priority.
    scores[0].zero_()
    if rows > 1:
        scores[1, 12:] = -torch.inf
    select = (
        indexer._stable_topk_indices if policy == indexer.QSA_SELECTION_STABLE_ARGSORT else indexer._fast_topk_indices
    )
    expected = select(scores, k)
    calls = []

    def bounded_select(tile, count):
        calls.append(tile.shape)
        assert tile.numel() <= width * 7
        return select(tile, count)

    monkeypatch.setattr(indexer, "_QSA_SELECTION_MAX_TILE_ELEMENTS", width * 7)
    name = "_stable_topk_indices" if policy == indexer.QSA_SELECTION_STABLE_ARGSORT else "_fast_topk_indices"
    monkeypatch.setattr(indexer, name, bounded_select)
    actual = indexer._select_topk_indices(scores, k, policy)
    assert torch.equal(actual, expected)
    assert len(calls) == (rows + 6) // 7


@pytest.mark.parametrize("policy", sorted(indexer.QSA_SELECTION_POLICIES))
@pytest.mark.parametrize("lengths", [(23,), (0, 23, 0), (9, 0, 14)])
@pytest.mark.parametrize("ties", [False, True])
def test_streamed_scores_preserve_request_mapping_mask_and_selection(monkeypatch, policy, lengths, ties):
    rows, heads, dim = sum(lengths), 4, 8
    groups_per_block, table_width, k = 4, 7, 5
    capacity = groups_per_block * table_width
    generator = torch.Generator().manual_seed(29)
    query = torch.randint(-2, 3, (rows, heads, dim), generator=generator).half()
    cache = torch.randint(-2, 3, (21, groups_per_block + 3, dim), generator=generator).half()
    if ties:
        query.zero_()
    # Requests address different physical pages; scratch rows are poisoned.
    cache[:, groups_per_block:] = 30000
    table = torch.stack([torch.arange(table_width) + request * table_width for request in range(len(lengths))])
    visible = torch.tensor([(row * 3) % (capacity + 1) for row in range(rows)], dtype=torch.int32)
    expected_parts = []
    start = 0
    for request, length in enumerate(lengths):
        if length:
            keys = cache.index_select(0, table[request])[:, :groups_per_block].reshape(capacity, dim).float().t()
            scores = torch.matmul(query[start : start + length].float(), keys).relu_().sum(dim=1)
            scores.masked_fill_(
                torch.arange(capacity).unsqueeze(0) >= visible[start : start + length].unsqueeze(1), -torch.inf
            )
            chosen = indexer._select_topk_indices(scores, k, policy).to(torch.int32)
            expected_parts.append(indexer._repair_native_group_indices(chosen, visible[start : start + length]))
        start += length
    expected = torch.cat(expected_parts)
    original_select = indexer._select_topk_indices
    tile_shapes = []

    def bounded_select(scores, count, selected_policy):
        tile_shapes.append(scores.shape)
        assert scores.numel() <= capacity * 3
        return original_select(scores, count, selected_policy)

    def forbid_concat(*args, **kwargs):
        raise AssertionError("prefill score tiles must not be concatenated")

    monkeypatch.setattr(indexer, "_QSA_SELECTION_MAX_TILE_ELEMENTS", capacity * 3)
    monkeypatch.setattr(indexer, "_select_topk_indices", bounded_select)
    monkeypatch.setattr(torch, "cat", forbid_concat)
    actual = indexer._qsa_matmul_select_groups(query, cache, table, visible, lengths, groups_per_block, k, policy)
    assert actual.dtype == torch.int32
    assert torch.equal(actual, expected)
    assert sum(shape[0] for shape in tile_shapes) == rows


def test_production_256k_geometry_bounds_selection_to_64_rows():
    capacity = 262144 // 4
    assert indexer._QSA_SELECTION_MAX_TILE_ELEMENTS // capacity == 64
    # The score tensor is <=16 MiB, versus about 552 MiB for the crashing
    # 2,560-row cached-prefix prefill. Backend workspace is measured on NPU.
    assert indexer._QSA_SELECTION_MAX_TILE_ELEMENTS * 4 == 16 * 1024 * 1024
