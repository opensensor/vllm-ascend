# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission and exact-selection gates; no NPU speed claims."""

from dataclasses import replace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qsa_graph_selection as candidate
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    QSAGroupSelection,
    _qsa_position_geometry,
    _repair_native_group_indices,
    _select_topk_indices,
    qsa_indexer_score_310_reference,
)


def plan(length=512, queries=(3,), **kwargs):
    return candidate.make_plan([length] * len(queries), queries, table_blocks=4096, groups_per_block=16, **kwargs)


@pytest.mark.parametrize("length", [0, 3, 4, 63, 64, 65, 2048, 2051, 2052, 4096, 8000, 8128, 8192, 262144])
@pytest.mark.parametrize("queries", [(1,), (3,), (3, 3), (0, 3)])
def test_plan_admits_original_input_and_never_changes_score_backend(length, queries):
    captured = plan(length, queries, max_matmul_decode_tokens=8)
    assert captured.admits([length] * len(queries), queries)
    assert captured.capacity_blocks <= 4096
    assert captured.capacity_blocks >= max(1, ((length // 4) + 15) // 16)
    assert not captured.admits([length] * len(queries), tuple(q + 1 for q in queries))
    if captured.capacity_groups * 4 + 4 <= captured.max_context:
        assert not captured.admits([captured.capacity_groups * 4 + 4] * len(queries), queries)


def test_prefix_and_score_backend_boundaries_require_new_capture():
    prefix = plan(2048)
    assert prefix.backend == "prefix"
    assert prefix.admits([2051], [3])
    assert not prefix.admits([2052], [3])
    native = plan(8000, max_matmul_decode_tokens=8)
    assert native.backend == "native"
    assert native.capacity_groups == 2032
    assert not native.admits([8192], [3])
    matrix = plan(8192, max_matmul_decode_tokens=8)
    assert matrix.backend == "matmul"
    assert not matrix.admits([4096], [3])
    assert plan(262144, max_matmul_decode_tokens=8).capacity_groups == 65536


@pytest.mark.parametrize("policy", ["stable_argsort", "fast_topk"])
@pytest.mark.parametrize("length", [0, 3, 4, 7, 8, 63, 64, 65, 511, 512, 2048, 2051])
def test_prefix_skips_scoring_but_preserves_every_consumed_group_and_causal_tail(length, policy):
    positions = torch.arange(length - 3, length, dtype=torch.int32)
    captured = plan(length, selection_policy=policy)
    result = candidate.prefix_selection(
        positions, compress_ratio=4, token_topk=2048, capacity_groups=captured.capacity_groups
    )
    actual_capacity = max(1, ((length // 4) + 15) // 16) * 16
    visible, tail_starts, tail_counts = _qsa_position_geometry(positions, 4, actual_capacity)
    scores = torch.randn(3, actual_capacity, generator=torch.Generator().manual_seed(42))
    scores.masked_fill_(torch.arange(actual_capacity)[None] >= visible[:, None], -torch.inf)
    width = min(512, actual_capacity)
    selected = _repair_native_group_indices(_select_topk_indices(scores, width, policy), visible)
    assert torch.equal(result.group_counts, visible.clamp_max(width))
    assert torch.equal(result.tail_starts, tail_starts)
    assert torch.equal(result.tail_counts, tail_counts)
    for row, count in enumerate(result.group_counts.tolist()):
        assert torch.equal(result.group_indices[row, : max(0, count)], selected[row, : max(0, count)])


@pytest.mark.parametrize("bad", [torch.tensor([1.5]), torch.ones((1, 1), dtype=torch.int32), [1.5], [True]])
def test_admission_rejects_noninteger_or_nonvector_lengths(bad):
    with pytest.raises(ValueError):
        candidate.make_plan(bad, [1], table_blocks=4096, groups_per_block=16)


def test_admission_never_reads_device_lengths():
    with pytest.raises(ValueError, match="scheduler CPU"):
        plan().admits(torch.empty(1, dtype=torch.int32, device="meta"), [3])


@pytest.mark.parametrize("length,queries", [(-1, [3]), (262145, [3]), (10, [0]), (10, [-1])])
def test_admission_fails_closed_for_invalid_batch(length, queries):
    with pytest.raises(ValueError):
        plan().admits([length], queries)


def test_partial_page_context_limit_requires_actual_table_capacity():
    with pytest.raises(ValueError, match="block table"):
        candidate.make_plan([262144], [3], table_blocks=4000, groups_per_block=16)


def test_prefix_uses_changing_logical_positions_and_rejects_mrope_axes():
    captured = plan(2048)
    query = torch.empty(3, 4, 128)
    cache = torch.empty(32, 19, 128)
    table = torch.empty(1, 4096, dtype=torch.int32)
    boundaries = torch.tensor([0, 3], dtype=torch.int32)
    for length in (65, 511, 2048, 0):
        positions = torch.arange(length - 3, length, dtype=torch.int32)
        result = candidate.select_with_plan(query, cache, table, boundaries, positions, plan=captured)
        assert torch.equal(result.group_counts, torch.div(positions + 1, 4, rounding_mode="floor"))
    with pytest.raises(ValueError, match="causal metadata"):
        candidate.select_with_plan(query, cache, table, boundaries, positions.expand(3, -1), plan=captured)


@pytest.mark.parametrize("policy", ["stable_argsort"])
def test_bucketed_native_score_consumes_dynamic_pages_and_preserves_exact_selected_order(monkeypatch, policy):
    # Small geometry exercises truncation, ties and noncontiguous physical
    # pages without pretending CPU reference tests qualify the NPU kernels.
    captured = candidate.make_plan(
        [15], [3], table_blocks=16, groups_per_block=2, token_topk=16, selection_policy=policy
    )
    assert captured.backend == "prefix"
    captured = candidate.make_plan(
        [43], [3], table_blocks=16, groups_per_block=2, token_topk=16, selection_policy=policy
    )
    assert captured.backend == "native"

    def reference(query, cache, table, starts, positions, **kwargs):
        blocks = (kwargs["max_visible_groups"] + 1) // 2
        scores = qsa_indexer_score_310_reference(query, cache, table[:, :blocks], starts, positions)
        visible, tails, counts = _qsa_position_geometry(positions, 4, blocks * 2)
        selected = _repair_native_group_indices(_select_topk_indices(scores, 4, kwargs["selection_policy"]), visible)
        return QSAGroupSelection(selected.int(), visible.clamp_max(4).int(), tails.int(), counts.int())

    monkeypatch.setattr(candidate, "qsa_indexer_select_groups_310", reference)
    query = torch.ones(3, 2, 8, dtype=torch.float16)
    # Unique integer scores avoid silently accepting a changed fast_topk tie
    # cutoff; stable policy also checks ties separately below.
    cache = torch.arange(16 * 5, dtype=torch.float16).view(16, 5, 1).expand(16, 5, 8).clone()
    starts = torch.tensor([0, 3], dtype=torch.int32)
    for length, physical in [
        (43, torch.arange(16)),
        (47, torch.arange(15, -1, -1)),
        (60, torch.roll(torch.arange(16), 3)),
    ]:
        assert captured.admits([length], [3])
        table = physical.int().unsqueeze(0)
        positions = torch.arange(length - 3, length, dtype=torch.int32)
        actual = candidate.select_with_plan(query, cache, table, starts, positions, plan=captured)
        expected = reference(
            query,
            cache,
            table,
            starts,
            positions,
            max_visible_groups=length // 4,
            selection_policy=policy,
        )
        assert torch.equal(actual.group_indices, expected.group_indices)
        assert torch.equal(actual.group_counts, expected.group_counts)
        assert torch.equal(actual.tail_starts, expected.tail_starts)
        assert torch.equal(actual.tail_counts, expected.tail_counts)
    if policy == "stable_argsort":
        cache.zero_()
        result = candidate.select_with_plan(query, cache, table, starts, positions, plan=captured)
        assert torch.equal(result.group_indices, torch.arange(4).expand(3, -1))


def test_select_fails_closed_on_changed_packed_geometry():
    captured = replace(plan(2048), query_lens=(1, 2))
    with pytest.raises(ValueError, match="query geometry"):
        candidate.select_with_plan(
            torch.empty(3, 4, 128),
            torch.empty(32, 19, 128),
            torch.empty(1, 4096),
            torch.tensor([0, 3]),
            torch.arange(3),
            plan=captured,
        )


def test_fast_topk_nonprefix_requires_exact_score_width_to_preserve_cutoff_ties():
    captured = plan(4300, selection_policy="fast_topk")
    assert captured.capacity_blocks == 68
    assert captured.admits([4303], [3])
    assert not captured.admits([4096], [3])
    assert not captured.admits([4356], [3])


def test_hardware_gate_checker_rejects_reordered_ids_and_changed_tail():
    from tools.qwen4exp.benchmark_qsa_graph_selection_310 import assert_same_selection

    expected = QSAGroupSelection(torch.tensor([[2, 1, -1]]), torch.tensor([2]), torch.tensor([12]), torch.tensor([1]))
    # Unused padding can differ; consumed IDs and their order cannot.
    assert_same_selection(replace(expected, group_indices=torch.tensor([[2, 1, 0]])), expected)
    with pytest.raises(AssertionError):
        assert_same_selection(replace(expected, group_indices=torch.tensor([[1, 2, -1]])), expected)
    with pytest.raises(AssertionError):
        assert_same_selection(replace(expected, tail_counts=torch.tensor([0])), expected)


@pytest.mark.parametrize("ratio,budget,capacity", [(0, 2048, 512), (4, 0, 512), (4, 2047, 512), (4, 2048, 513)])
def test_prefix_rejects_invalid_geometry(ratio, budget, capacity):
    with pytest.raises(ValueError):
        candidate.prefix_selection(torch.arange(3), compress_ratio=ratio, token_topk=budget, capacity_groups=capacity)
