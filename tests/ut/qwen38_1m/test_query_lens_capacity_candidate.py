# SPDX-License-Identifier: Apache-2.0
"""Expanded MTP qLens must retain an owned buffer and independent draft views."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates.query_lens_capacity import fill_query_lens_cpu


@pytest.mark.parametrize("drafting", [False, True])
def test_expanded_rows_do_not_resize_an_out_view(drafting):
    builder = SimpleNamespace(_query_lens_cpu_buffer=torch.empty(4, dtype=torch.int32))
    boundaries = torch.arange(13, dtype=torch.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fill_query_lens_cpu(builder, 12, boundaries, drafting)
    torch.testing.assert_close(result, torch.ones(12, dtype=torch.int32))
    assert builder._query_lens_cpu_buffer.numel() == 12
    assert (result.data_ptr() != builder._query_lens_cpu_buffer.data_ptr()) == drafting
    storage = builder._query_lens_cpu_buffer.data_ptr()
    next_result = fill_query_lens_cpu(builder, 4, torch.arange(5, dtype=torch.int32) * 3, drafting)
    assert builder._query_lens_cpu_buffer.data_ptr() == storage
    torch.testing.assert_close(next_result, torch.full((4,), 3, dtype=torch.int32))
    if drafting:
        torch.testing.assert_close(result, torch.ones(12, dtype=torch.int32))


def test_no_buffer_returns_contiguous_lengths():
    builder = SimpleNamespace(_query_lens_cpu_buffer=None)
    result = fill_query_lens_cpu(builder, 3, torch.tensor([0, 2, 5, 9], dtype=torch.int32))
    torch.testing.assert_close(result, torch.tensor([2, 3, 4], dtype=torch.int32))
    assert result.is_contiguous()


def test_growth_preserves_pinned_allocation_request(monkeypatch):
    class PinnedBuffer:
        def numel(self):
            return 4

        def is_pinned(self):
            return True

    original_empty = torch.empty
    allocations = []

    def allocate(*args, **kwargs):
        allocations.append(kwargs.copy())
        kwargs["pin_memory"] = False  # Host-only test has no NPU pin allocator.
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    builder = SimpleNamespace(_query_lens_cpu_buffer=PinnedBuffer())
    result = fill_query_lens_cpu(builder, 12, torch.arange(13, dtype=torch.int32))
    assert allocations == [{"dtype": torch.int32, "device": "cpu", "pin_memory": True}]
    torch.testing.assert_close(result, torch.ones(12, dtype=torch.int32))
