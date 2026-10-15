# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.qwen4exp.tiled_qsa_selection import tiled_selector


def test_stable_ties_and_partial_last_tile_preserve_indices_and_bound_work():
    scores = torch.tensor(
        [
            [1.0, 3.0, 3.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, 8.0, 8.0, 8.0, 1.0],
            [-1.0, 2.0, 4.0, 3.0, 0.0],
            [3.0, 1.0, 0.0, -1.0, 2.0],
        ]
    )
    calls = []

    def original(rows, k, policy):
        calls.append((rows.shape[0], policy))
        return rows.argsort(dim=1, descending=True, stable=True)[:, :k].to(torch.int32)

    actual = tiled_selector(original, 10)(scores, 3, "stable")
    expected = scores.argsort(dim=1, descending=True, stable=True)[:, :3].to(torch.int32)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.int32
    assert calls == [(2, "stable"), (2, "stable"), (1, "stable")]


def test_small_batch_preserves_original_output_identity():
    scores = torch.ones(3, 8)
    result = torch.zeros(3, 2, dtype=torch.int32)
    assert tiled_selector(lambda *args: result, 24)(scores, 2, "fast") is result


def test_invalid_budget():
    with pytest.raises(ValueError, match="positive"):
        tiled_selector(lambda *args: None, 0)
