# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare future graph decisions against the current scalar commit contract."""

import itertools

import pytest
import torch

from tools.qwen4exp.mamba_commit_plan import plan_aligned_commit


def reference(source, computed, scheduled, drafts, accepted, block_size):
    running = computed + scheduled - drafts
    new_computed = running + accepted - 1
    aligned = new_computed // block_size * block_size
    if aligned < running:
        return 0, 0, False, accepted
    destination = aligned // block_size - 1
    bias = aligned - running
    if source == destination:
        if bias == 0:
            return 0, 0, False, 1
        return destination, bias, True, 1
    return destination, bias, True, accepted


@pytest.mark.parametrize("block_size", [1, 4, 16, 128, 256])
def test_exhaustive_near_boundary_matches_scalar_decisions(block_size):
    rows = list(itertools.product(range(4), range(max(0, block_size - 3), block_size * 2 + 3), [1, 3], range(4)))
    rows = [(source, computed, scheduled, scheduled - 1, accepted) for source, computed, scheduled, accepted in rows]
    inputs = [torch.tensor(column, dtype=torch.int32) for column in zip(*rows)]
    originals = [value.clone() for value in inputs]
    actual = plan_aligned_commit(*inputs, block_size=block_size)
    expected = list(zip(*(reference(*row, block_size) for row in rows)))
    for value, column in zip(actual, expected):
        assert value.tolist() == list(column)
    for value, before in zip(inputs, originals):
        assert torch.equal(value, before)


def test_changed_acceptance_updates_plan_without_host_value_reads(monkeypatch):
    inputs = [torch.tensor(column, dtype=torch.int32) for column in ([0, 1], [126, 255], [3, 3], [2, 2], [1, 3])]

    def forbidden(*args, **kwargs):
        raise AssertionError("no device-to-host observation allowed")

    with monkeypatch.context() as patch:
        for name in ("item", "tolist", "cpu", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        first = plan_aligned_commit(*inputs, block_size=128)
        inputs[-1].copy_(torch.tensor([3, 1], dtype=torch.int32))
        second = plan_aligned_commit(*inputs, block_size=128)
    assert first.copy_required.tolist() == [False, False]
    assert second.copy_required.tolist() == [True, False]
    assert second.accepted_token_bias.tolist() == [1, 0]


def test_full_context_positions_stay_exact():
    rows = [(2046, 262142, 3, 2, 3), (2047, 262143, 3, 2, 1)]
    inputs = [torch.tensor(column, dtype=torch.int32) for column in zip(*rows)]
    actual = plan_aligned_commit(*inputs, block_size=128)
    assert [value.tolist() for value in actual] == [list(column) for column in zip(*(reference(*r, 128) for r in rows))]


@pytest.mark.parametrize("block_size", [0, -1, True, 1 << 31])
def test_bad_block_size_rejected(block_size):
    values = [torch.ones(2, dtype=torch.int32) for _ in range(5)]
    with pytest.raises(ValueError):
        plan_aligned_commit(*values, block_size=block_size)


def test_shape_and_dtype_contracts():
    values = [torch.ones(2, dtype=torch.int32) for _ in range(5)]
    for bad in (
        torch.ones(2, dtype=torch.int64),
        torch.ones(3, dtype=torch.int32),
        torch.ones(1, 2, dtype=torch.int32),
    ):
        with pytest.raises(ValueError):
            plan_aligned_commit(*values[:4], bad, block_size=128)
