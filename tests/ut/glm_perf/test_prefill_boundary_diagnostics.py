# SPDX-License-Identifier: Apache-2.0

import io
import json

import pytest
import torch

from tools.glm_perf.trace_device_ops import DeviceOpTrace, tensor_metadata
from vllm_ascend.models.glm5next.kpool_ops import topk_pool_indices


@pytest.mark.parametrize("width", [4096, 4160, 4320, 4352, 77760])
@pytest.mark.parametrize("rows", [1, 8, 33])
def test_row_bounded_selection_preserves_exact_indices(width, rows):
    generator = torch.Generator().manual_seed(77)
    # Unique scores; sliced storage exercises nonzero offsets and strides.
    scores = torch.stack([torch.randperm(width + 2, generator=generator) for _ in range(rows)]).float()[:, 1:-1]
    expected = topk_pool_indices(scores, 512)
    actual = topk_pool_indices(scores, 512, rows_per_call=32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_row_bounded_selection_ties_and_partial_last_batch():
    scores = torch.zeros(640, 4320)
    scores[:, 4000:] = -torch.inf
    torch.testing.assert_close(
        topk_pool_indices(scores, 512, rows_per_call=32), topk_pool_indices(scores, 512), rtol=0, atol=0
    )
    torch.testing.assert_close(
        topk_pool_indices(scores[:639], 512, rows_per_call=32), topk_pool_indices(scores[:639], 512), rtol=0, atol=0
    )


def test_row_bounded_dispatch_limits_rows_without_partitioning_columns(monkeypatch):
    original = torch.topk
    shapes = []

    def audited(value, *args, **kwargs):
        shapes.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", audited)
    topk_pool_indices(torch.randn(65, 4320), 512, rows_per_call=32)
    assert shapes == [(32, 4320), (32, 4320), (1, 4320)]
    with pytest.raises(ValueError, match="positive"):
        topk_pool_indices(torch.empty(1, 4320), 512, rows_per_call=0)


def test_trace_records_storage_views_without_tensor_contents():
    base = torch.arange(50).reshape(5, 10)
    view = base[1:, 2::2]
    meta = tensor_metadata(view)
    assert meta["storage_address"] == hex(base.untyped_storage().data_ptr())
    assert meta["storage_bytes"] == base.untyped_storage().nbytes()
    assert meta["storage_offset"] == 12
    assert meta["stride"] == [10, 2]
    assert tensor_metadata("private prompt") == {"type": "str"}


def test_trace_marks_boundary_and_completion_in_order():
    output = io.StringIO()
    syncs = []
    value = torch.ones(2)
    with DeviceOpTrace(output, lambda: syncs.append(True)):
        result = value + 1
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [event["event"] for event in events] == ["boundary_begin", "boundary_end", "begin", "submitted", "end"]
    assert events[2]["op"] == "aten.add.Tensor"
    assert len(syncs) == 2
    torch.testing.assert_close(result, torch.full((2,), 2.0))


def test_trace_distinguishes_previous_work_failure_from_current_op():
    def failed_sync():
        raise RuntimeError("device failed")

    output = io.StringIO()
    with pytest.raises(RuntimeError, match="device failed"), DeviceOpTrace(output, failed_sync):
        pytest.fail("body must not run after boundary failure")
    assert [json.loads(line)["event"] for line in output.getvalue().splitlines()] == ["boundary_begin"]


def test_trace_records_sync_failure_without_claiming_completion():
    calls = []

    def failed_second_sync():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("device failed")

    output = io.StringIO()
    value = torch.ones(2)
    with pytest.raises(RuntimeError, match="device failed"), DeviceOpTrace(output, failed_second_sync):
        value + 1
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [event["event"] for event in events][-2:] == ["submitted", "error"]
    assert events[-1]["error_type"] == "RuntimeError"
