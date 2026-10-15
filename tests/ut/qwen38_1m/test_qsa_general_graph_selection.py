# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU semantics for padding safety; these do not qualify NPU graph capture."""

from dataclasses import replace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qsa_general_graph_selection as candidate
from tools.qwen4exp.resident_candidates.qsa_graph_selection import make_plan
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import _qsa_matmul_select_groups, _qsa_position_geometry


def make_matmul_plan(policy="stable_argsort"):
    return make_plan(
        [8192, 0, 10240],
        [3, 0, 3],
        table_blocks=4096,
        groups_per_block=16,
        max_matmul_decode_tokens=8,
        selection_policy=policy,
    )


@pytest.mark.parametrize("policy", ["stable_argsort", "fast_topk"])
def test_unused_poisoned_pages_are_ignored_with_exact_consumed_order(policy):
    plan = make_matmul_plan(policy)
    assert plan.backend == "matmul"
    table = torch.arange(plan.capacity_blocks, dtype=torch.int32).remainder(64).expand(3, -1).clone()
    clean = table.clone()
    # Unallocated positive IDs would fault the original gather; negative padding
    # and a zero-query request must also never affect selection or the caller.
    table[0, 128:] = 1000000
    table[1] = 1000000
    table[2, 160:] = -1
    before = table.clone()
    cache = torch.arange(64 * 19, dtype=torch.float32).view(64, 19, 1).expand(64, 19, 8).clone()
    query = torch.ones(6, 4, 8)
    for lengths in ([8192, 10240], [8189, 10239]):
        assert plan.admits([lengths[0], 0, lengths[1]], [3, 0, 3])
        positions = torch.cat([torch.arange(length - 3, length) for length in lengths]).int()
        safe = candidate.mask_unused_pages(table, positions, plan=plan)
        visible, _, _ = _qsa_position_geometry(positions, 4, plan.capacity_groups)
        actual = _qsa_matmul_select_groups(query, cache, safe, visible, plan.query_lens, 16, 512, policy)
        expected = _qsa_matmul_select_groups(query, cache, clean, visible, plan.query_lens, 16, 512, policy)
        assert torch.equal(actual, expected)
        assert torch.equal(table, before)
        assert torch.count_nonzero(safe[1]) == 0
        assert torch.count_nonzero(safe[0, 128:]) == 0
        assert torch.equal(safe[0, :127], table[0, :127])


def test_live_bad_page_is_preserved_and_gather_still_rejects_it():
    plan = make_matmul_plan()
    table = torch.zeros((3, plan.capacity_blocks), dtype=torch.int32)
    table[0, 2] = 1000000
    positions = torch.tensor([8189, 8190, 8191, 10237, 10238, 10239], dtype=torch.int32)
    safe = candidate.mask_unused_pages(table, positions, plan=plan)
    assert safe[0, 2] == 1000000
    with pytest.raises(IndexError):
        torch.index_select(torch.zeros(64, 19, 8), 0, safe[0])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_causal_partial_groups_padded_rows_and_table_aliasing(dtype):
    plan = replace(make_matmul_plan(), query_lens=(3, 0, 3), capacity_blocks=4, groups_per_block=2)
    table = torch.tensor([[7, 8, -1, 1000], [1000] * 4, [1000] * 4], dtype=dtype)
    for positions, used in [([6, 7, 8, -3, -2, -1], 1), ([10, 11, 12, -3, -2, -1], 2)]:
        positions = torch.tensor(positions, dtype=dtype)
        safe = candidate.mask_unused_pages(table, positions, plan=plan)
        assert safe.dtype == dtype
        assert safe.data_ptr() != table.data_ptr()
        assert torch.equal(safe[0, :used], table[0, :used])
        assert torch.count_nonzero(safe[0, used:]) == 0
        assert torch.count_nonzero(safe[1:]) == 0
    assert table[0, 2] == -1


def test_dynamic_extent_does_not_read_device_values_in_python():
    plan = make_matmul_plan()
    safe = candidate.mask_unused_pages(
        torch.empty((3, plan.capacity_blocks), dtype=torch.int32, device="meta"),
        torch.empty(6, dtype=torch.int32, device="meta"),
        plan=plan,
    )
    assert safe.shape == (3, plan.capacity_blocks)


@pytest.mark.parametrize("backend", ["prefix", "native", "matmul"])
def test_wrapper_only_masks_matmul_and_preserves_plan(monkeypatch, backend):
    plan = replace(make_matmul_plan(), backend=backend)
    table = torch.full((3, plan.capacity_blocks), 1000000, dtype=torch.int32)
    positions = torch.full((6,), -1, dtype=torch.int32)
    seen = []

    def select(query, cache, current_table, starts, current_positions, *, plan):
        seen.append((current_table, current_positions, plan))
        return "selected"

    monkeypatch.setattr(candidate, "select_with_plan", select)
    assert candidate.select_general_with_plan(None, None, table, None, positions, plan=plan) == "selected"
    current_table, current_positions, current_plan = seen[0]
    assert current_plan is plan and current_positions is positions
    if backend == "matmul":
        assert current_table is not table
        assert torch.count_nonzero(current_table) == 0
    else:
        assert current_table is table


@pytest.mark.parametrize("failure", ["requests", "capacity", "positions", "dtype", "device"])
def test_static_geometry_errors_fail_before_device_work(failure):
    plan = make_matmul_plan()
    table = torch.empty((3, plan.capacity_blocks), dtype=torch.int32)
    positions = torch.empty(6, dtype=torch.int32)
    if failure == "requests":
        table = table[:2]
    elif failure == "capacity":
        table = table[:, :-1]
    elif failure == "positions":
        positions = positions[:5]
    elif failure == "dtype":
        positions = positions.float()
    else:
        positions = positions.to("meta")
    with pytest.raises(ValueError):
        candidate.mask_unused_pages(table, positions, plan=plan)
