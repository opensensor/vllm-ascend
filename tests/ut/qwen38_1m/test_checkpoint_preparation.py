# SPDX-License-Identifier: Apache-2.0
import itertools

import pytest
import torch

from tools.qwen4exp.checkpoint_preparation import prepare_checkpoint
from tools.qwen4exp.mamba_commit import PADDED_ROWS, reference


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("computed", [0, 1, 124, 125, 126, 127, 128, 129, 253, 254, 255, 256, 8190])
@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_device_decision_matches_scalar_precopy(n, computed, accepted):
    current = (computed + 2) // 128
    previous = torch.tensor(
        list(itertools.islice(itertools.cycle([-1, current, max(0, current - 1), current + 1]), n)), dtype=torch.int32
    )
    inputs = torch.zeros((5, PADDED_ROWS), dtype=torch.int32)
    result = prepare_checkpoint(
        previous,
        torch.full((n,), computed, dtype=torch.int32),
        torch.full((n,), accepted, dtype=torch.int32),
        inputs,
        block_size=128,
    )
    assert not result.error.any()
    columns = current + 5
    table = torch.arange(columns, dtype=torch.int32).expand(n, -1).contiguous()
    # Explicit sequential scalar oracle; arbitrary bits verify complete state
    # preservation and convolution tail behavior, including overlapping blocks.
    for conv in (False, True):
        state = torch.arange(columns * 5 * 8, dtype=torch.int32).reshape(columns, 5, 8)
        expected = state.clone()
        for row, prev in enumerate(previous.tolist()):
            copy = prev != -1 and prev != current
            assert result.copy_required[row] == copy
            assert result.state_indices[row] == current
            assert result.normalized_counts[row] == (1 if copy else accepted)
            if copy:
                selected = expected[prev, accepted - 1 :] if conv else expected[prev + accepted - 1]
                expected[current].reshape(-1)[: selected.numel()].copy_(selected.reshape(-1).clone())
        actual, receipt = reference(state, inputs, table, conv=conv, block_size=128)
        assert not receipt[1].any()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("previous,computed,accepted", [(-2, 10, 1), (0, -1, 1), (0, 126, 0), (0, 126, 4)])
def test_invalid_values_cannot_write_banks(previous, computed, accepted):
    inputs = torch.zeros((5, PADDED_ROWS), dtype=torch.int32)
    result = prepare_checkpoint(
        *(torch.tensor([value], dtype=torch.int32) for value in (previous, computed, accepted)), inputs, block_size=128
    )
    assert result.error.all() and not result.copy_required.any()
    state = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    actual, receipt = reference(
        state, inputs, torch.arange(4, dtype=torch.int32).unsqueeze(0), conv=False, block_size=128
    )
    assert not receipt[1].any()
    torch.testing.assert_close(actual, state, rtol=0, atol=0)


def test_wrong_dtype_rejected_before_submission():
    value = torch.ones(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="INT32"):
        prepare_checkpoint(value, value, value, torch.zeros((5, PADDED_ROWS), dtype=torch.int32), block_size=128)
