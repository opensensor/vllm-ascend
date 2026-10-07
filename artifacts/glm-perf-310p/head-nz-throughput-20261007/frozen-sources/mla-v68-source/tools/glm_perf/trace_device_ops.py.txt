# SPDX-License-Identifier: Apache-2.0
"""Opt-in operator attribution for a bounded diagnostic forward.

Enter DeviceOpTrace around the suspect eager prefill section, outside graph
capture. Each operation is logged before dispatch and synchronized afterward.
The final unmatched begin record identifies the operation whose completion
stalled. This deliberately changes scheduling and can hide lifetime races;
absence of a failure under this trace is not a correctness result.

No tensor contents are read and no NPU is initialized on import. The caller
owns the log and supplies the synchronization callback. Never enable this
across production forwards or graph capture.
"""

import json
from collections.abc import Callable
from typing import IO

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_metadata(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "storage_offset": value.storage_offset(),
            "storage_bytes": value.untyped_storage().nbytes(),
            "storage_address": hex(value.untyped_storage().data_ptr()),
        }
    if isinstance(value, (tuple, list)):
        return [tensor_metadata(child) for child in value]
    if isinstance(value, dict):
        return {str(key): tensor_metadata(child) for key, child in value.items()}
    # Exclude scalar/string values: tracing should not log request contents.
    return {"type": type(value).__name__}


class DeviceOpTrace(TorchDispatchMode):
    def __init__(self, output: IO[str], synchronize: Callable[[], None]):
        super().__init__()
        self.output = output
        self.synchronize = synchronize
        self.sequence = 0

    def emit(self, **record):
        self.output.write(json.dumps(record) + "\n")
        self.output.flush()

    def __enter__(self):
        # Attribute pending work to the boundary, never to the first traced op.
        self.emit(event="boundary_begin")
        self.synchronize()
        self.emit(event="boundary_end")
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.sequence += 1
        sequence = self.sequence
        self.emit(event="begin", sequence=sequence, op=str(func), inputs=tensor_metadata((args, kwargs or {})))
        try:
            result = func(*args, **(kwargs or {}))
            self.emit(event="submitted", sequence=sequence, outputs=tensor_metadata(result))
            self.synchronize()
        except Exception as error:
            self.emit(event="error", sequence=sequence, error_type=type(error).__name__)
            raise
        self.emit(event="end", sequence=sequence)
        return result
