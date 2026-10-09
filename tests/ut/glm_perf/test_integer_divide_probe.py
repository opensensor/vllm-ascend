# SPDX-License-Identifier: Apache-2.0
"""Verify actual probe reports against the stricter v2 admission contract."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.integer_divide import DIVISION_GATE_COUNTS, DIVISORS
from tools.glm_perf.integer_divide_probe import case


@pytest.mark.parametrize("count", DIVISION_GATE_COUNTS)
@pytest.mark.parametrize("divisor", DIVISORS)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_tiny_and_large_probe_cases_report_signed_extremes(monkeypatch, count, divisor, dtype):
    state = SimpleNamespace(capturing=None)

    class Graph:
        def replay(self):
            source, output, denominator = self.operation
            output.copy_(torch.div(source, denominator, rounding_mode="floor"))

    class Capture:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            state.capturing = self.graph

        def __exit__(self, *_):
            state.capturing = None

    class Native:
        device = torch.device("cpu")

        def __call__(self, source, denominator):
            backing = torch.zeros(source.numel() + 8, dtype=source.dtype)
            output = backing[: source.numel()]
            output.copy_(torch.div(source, denominator, rounding_mode="floor"))
            if state.capturing is not None:
                state.capturing.operation = source, output, denominator
            return output

    monkeypatch.setattr(torch, "npu", SimpleNamespace(NPUGraph=Graph, graph=Capture), raising=False)
    report = case(Native(), dtype, divisor, count)
    assert report["passed"] and report["signed_extremes"] and report["changed_input_replay"]
    assert report["owned_padding_checked"]
