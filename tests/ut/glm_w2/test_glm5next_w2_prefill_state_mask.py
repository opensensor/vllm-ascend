# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the 310P KDA prefill state mask."""

from math import prod

import pytest
import torch

from vllm_ascend.models.glm5next_w2.kda_310 import _prefill_initial_state


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("state_shape", [(5, 2, 3), (5, 2, 3, 4)])
def test_prefill_state_mask_matches_boolean_assignment_without_index_put(monkeypatch, dtype, state_shape):
    state = torch.arange(prod(state_shape), dtype=torch.float32).reshape(state_shape).to(dtype)
    state[0].fill_(float("nan"))
    state[4].fill_(float("inf"))
    original = state.clone()
    indices = torch.tensor([1, 0, 4], dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, False])
    expected = state[indices].float().contiguous()
    expected[~has_initial_state] = 0

    original_setitem = torch.Tensor.__setitem__

    def reject_boolean_assignment(self, key, value):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            raise AssertionError("KDA prefill reintroduced dynamic boolean assignment")
        return original_setitem(self, key, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", reject_boolean_assignment)
    actual = _prefill_initial_state(state, indices, has_initial_state)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(state, original, equal_nan=True)
    assert actual.dtype == torch.float32
    assert torch.count_nonzero(actual[1:]) == 0
