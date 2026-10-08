# SPDX-License-Identifier: Apache-2.0
"""Authoritative source must select the query cast without trusting old wrappers."""

from types import FunctionType

import pytest
import torch

from tools.glm_perf.query_cast_binding import bounded_converter, wrap_forward


def original(self, q):
    return q


@pytest.mark.parametrize("expression", ["q.to(torch.bfloat16)", "q.to(dtype=torch.bfloat16)"])
def test_query_cast_is_replaced_even_when_previous_method_source_is_unrelated(expression):
    source = f"class Indexer:\n    def forward(self, q):\n        q = {expression}\n        return q\n"
    calls = []

    def converter(value, dtype):
        calls.append((value, dtype))
        return value.to(dtype)

    wrapped = wrap_forward(original, converter, source)
    value = torch.tensor([1.00390625, -0.0, -1.00390625, 123.4], dtype=torch.float16)
    assert torch.equal(wrapped(None, value).view(torch.int16), value.bfloat16().view(torch.int16))
    assert calls[0][0] is value and calls[0][1] is torch.bfloat16
    assert "_aicore_convert" in wrapped.__code__.co_names


@pytest.mark.parametrize(
    "body",
    [
        "return q",
        "q = q.to(torch.float16)\n        return q",
        "q = q.to(torch.bfloat16)\n        q = q.to(torch.bfloat16)\n        return q",
    ],
)
def test_missing_changed_or_duplicate_cast_refuses_partial_replacement(body):
    with pytest.raises(ValueError, match="query"):
        wrap_forward(original, None, "class Indexer:\n    def forward(self,q):\n        " + body)


@pytest.mark.parametrize("count", [2, 8, 9, 640])
def test_large_prefill_avoids_the_scalar_converter(count):
    calls = []

    def native(value, dtype):
        calls.append(value.numel())
        return value.to(dtype)

    value = torch.arange(count, dtype=torch.float16)
    actual = bounded_converter(native, 8)(value, torch.bfloat16)
    assert torch.equal(actual.view(torch.int16), value.bfloat16().view(torch.int16))
    assert calls == ([count] if count <= 8 else [])


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_query_cast_bound_is_rejected(limit):
    with pytest.raises(ValueError, match="positive"):
        bounded_converter(None, limit)


def test_query_rebinding_preserves_permanent_rope_conversions():
    source = """class Indexer:
    def forward(self, q, k_pe):
        k_pe = k_pe.to(torch.bfloat16)
        k_pe = k_pe.reshape(-1, 1, 2).float()
        q = q.to(torch.bfloat16)
        return q, k_pe
"""
    query_calls, legacy_calls = [], []

    def legacy(value, dtype):
        legacy_calls.append(dtype)
        return value.to(dtype)

    def query(value, dtype):
        query_calls.append(dtype)
        return value.to(dtype)

    bound = FunctionType(original.__code__, dict(original.__globals__, _aicore_convert=legacy))
    wrapped = wrap_forward(bound, query, source)
    q, k = torch.randn(4).half(), torch.randn(4)
    actual_q, actual_k = wrapped(None, q, k)
    assert torch.equal(actual_q, q.bfloat16())
    assert torch.equal(actual_k, k.bfloat16().reshape(-1, 1, 2).float())
    assert query_calls == [torch.bfloat16]
    assert legacy_calls == [torch.bfloat16, torch.float32]
