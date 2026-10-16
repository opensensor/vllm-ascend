# SPDX-License-Identifier: Apache-2.0
"""Private HC mix replay with explicit storage and stream ownership.

Capture is performed by the drained hardware gate, never lazily in serving.
The caller retains the record (graph, pool, inputs, outputs and stream).
"""

import torch


def matches(value, record, runtime):
    staged = record["input"]
    return (
        not torch.is_grad_enabled()
        and not runtime.is_current_stream_capturing()
        and runtime.current_stream(value.device) == record["stream"]
        and value.shape == staged.shape
        and value.dtype == staged.dtype
        and value.device == staged.device
        and value.is_contiguous()
        and not record.get("failed", False)
    )


def replay(value, record):
    record["input"].copy_(value)
    record["graph"].replay()
    mixed, (_, normalized) = record["output"]
    # Both tensors can outlive the next invocation. The residual must alias the
    # caller's input, not the static staging allocation overwritten by replay.
    return mixed.clone(), (value, normalized.clone())


class HCMixGraph:
    def __init__(self, original, record, runtime=None):
        if not record.get("qualified", False) or record.get("failed", False):
            raise ValueError("qualify the private graph before binding")
        self.original = original
        self.record = record
        self.runtime = torch.npu if runtime is None else runtime
        self.calls = 0
        self.fallbacks = 0

    def __call__(self, value):
        if not matches(value, self.record, self.runtime):
            self.fallbacks += 1
            return self.original(value)
        self.calls += 1
        return replay(value, self.record)
