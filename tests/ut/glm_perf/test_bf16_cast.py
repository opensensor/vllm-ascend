# SPDX-License-Identifier: Apache-2.0
"""BF16 conversion plumbing and focused source-patch behavior."""

import pytest
import torch

from tools.glm_perf.bf16_cast import NativeBF16Cast
from tools.glm_perf.resident_candidates.indexer_aicore_casts import wrap_indexer


def indexer_fixture(self, q, k_pe):
    q = q.to(torch.bfloat16)
    k_pe = k_pe.to(torch.bfloat16)
    k_pe = k_pe.reshape(-1, 1, self.rope_dim).float()
    return q, k_pe


def test_cpu_cast_fallback_preserves_shapes_special_values_and_strides():
    native = object.__new__(NativeBF16Cast)
    values = torch.tensor([[0.0, -0.0, 1.00390625, torch.inf], [-1.00390625, 70000.0, 1e-30, -torch.inf]])[:, ::2]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        actual = native(values, dtype)
        expected = values.to(dtype)
        assert torch.equal(actual, expected)
        assert actual.shape == values.shape
    assert native(values, values.dtype) is values


def test_indexer_rewrite_preserves_rounding_and_unwraps_previous_generation():
    calls = []

    def convert(value, dtype):
        calls.append(dtype)
        return value.to(dtype)

    class State:
        rope_dim = 2

    q = torch.tensor([1.00390625, -1.00390625])
    k = torch.tensor([[1.00390625, 1.01171875]])
    first = wrap_indexer(indexer_fixture, convert)
    wrapped = wrap_indexer(first, convert)
    actual = wrapped(State(), q, k)
    expected = indexer_fixture(State(), q, k)
    assert all(torch.equal(x, y) for x, y in zip(actual, expected))
    assert calls == [torch.bfloat16, torch.bfloat16, torch.float32]
    assert wrapped.__glm_aicore_original__ is indexer_fixture


def changed_indexer(self, q):
    return q.half()


def test_indexer_rewrite_rejects_changed_source_contract():
    with pytest.raises(ValueError, match="query conversion changed"):
        wrap_indexer(changed_indexer, lambda value, dtype: value.to(dtype))
