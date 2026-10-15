# SPDX-License-Identifier: Apache-2.0
"""Replay explicitly captured WY preparation with unchanged tensor arithmetic.

Capture and private qualification happen while the resident engine is drained.
The caller owns the graph and all static storage for the worker lifetime. A
single stream orders staging, replay, output copies and later buffer reuse.
Unsupported geometry, streams and enclosing captures retain the original path.
"""

import torch

CHUNK_SIZE = 64
OUTPUT_NAMES = ("q", "k", "w", "u", "g")


def matches(values, record, chunk_size):
    return (
        chunk_size == CHUNK_SIZE
        and len(values) == len(record.inputs) == len(OUTPUT_NAMES)
        and all(
            isinstance(value, torch.Tensor)
            and value.shape == staged.shape
            and value.dtype == staged.dtype
            and value.device == staged.device
            for value, staged in zip(values, record.inputs)
        )
    )


class WYGraph:
    def __init__(self, record, *, runtime=None):
        if record.failed or not record.receipt["passed"]:
            raise ValueError("WY replay requires a qualified, healthy private graph")
        self.record = record
        self.original = record.original
        self.runtime = torch.npu if runtime is None else runtime
        self.enabled = self.verify = True
        self.failed = False
        self.replays = self.checked = self.fallbacks = 0
        self.mismatched_outputs = []

    def __call__(self, q, k, v, g, beta, chunk_size):
        values = (q, k, v, g, beta)
        if (
            not self.enabled
            or self.failed
            or not matches(values, self.record, chunk_size)
            or self.runtime.is_current_stream_capturing()
            or self.runtime.current_stream(q.device) != self.record.replay_stream
        ):
            self.fallbacks += 1
            return self.original(*values, chunk_size)
        for staged, value in zip(self.record.inputs, values):
            staged.copy_(value)
        self.record.graph.replay()
        # Consumers can outlive this callback. Do not expose static graph
        # outputs which a later layer or queued iteration will overwrite.
        output = tuple(value.clone() for value in self.record.outputs)
        self.replays += 1
        if self.verify:
            expected = self.original(*values, chunk_size)
            for name, actual, reference in zip(OUTPUT_NAMES, output, expected):
                # Explicit qualification only; normal replay has no readback.
                if not torch.equal(actual.view(torch.int16), reference.view(torch.int16)):
                    self.failed = True
                    self.mismatched_outputs.append(name)
            self.checked += 1
            # Keep all ranks on reference outputs during shadow verification.
            return expected
        return output

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            replays=self.replays,
            checked=self.checked,
            fallbacks=self.fallbacks,
            mismatched_outputs=self.mismatched_outputs,
            input_shapes=[list(value.shape) for value in self.record.inputs],
            static_tensor_bytes=sum(
                value.numel() * value.element_size() for value in (*self.record.inputs, *self.record.outputs)
            ),
        )
