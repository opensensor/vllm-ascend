# SPDX-License-Identifier: Apache-2.0
"""Bounded direct adapter for the experimental 60-route INT4 down kernel.

Register a versioned binary before construction. This module never changes
OPP precedence or weights; each device/shape owns its tiling and workspace.
"""

import torch

INPUTS = 640
OUTPUTS = 2560
GROUP_SIZE = 128
METADATA_LANES = 8
MAX_ROUTES = 60
WORKSPACE_BYTES = 16 * 1024 * 1024
AI_CORE_BLOCKS = 8


def validate_geometry(arguments):
    if len(arguments) != 10:
        raise ValueError("direct down-reduce requires ten inputs")
    low, high, scales, sums, codes, weight_scales, offsets, weight_sums, routes, weights = arguments
    dtypes = (
        torch.int8,
        torch.int8,
        torch.float32,
        torch.float32,
        torch.int8,
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        torch.float32,
    )
    if any(t.dtype != dtype for t, dtype in zip(arguments, dtypes)):
        raise ValueError("direct down-reduce input dtypes differ from the binary")
    if any(t.device != low.device or not t.is_contiguous() for t in arguments):
        raise ValueError("direct down-reduce requires contiguous inputs on one device")
    if low.ndim != 2 or codes.ndim != 3 or routes.ndim != 1 or weights.ndim != 2:
        raise ValueError("invalid direct down-reduce ranks")
    rows, width = low.shape
    experts, columns, weight_width = codes.shape
    tokens, top_k = weights.shape
    if (
        not 0 < rows <= MAX_ROUTES
        or tokens <= 0
        or top_k <= 0
        or rows != tokens * top_k
        or routes.numel() != rows
        or width != INPUTS // 2
        or weight_width != width
        or columns != OUTPUTS
        or experts <= 0
    ):
        raise ValueError("unsupported direct down-reduce geometry")
    if (
        high.shape != low.shape
        or scales.shape != (rows, INPUTS // GROUP_SIZE, METADATA_LANES)
        or sums.shape != scales.shape
        or any(t.shape != (experts, OUTPUTS, INPUTS // GROUP_SIZE) for t in (weight_scales, offsets, weight_sums))
    ):
        raise ValueError("direct down-reduce metadata shape mismatch")
    return rows, experts, OUTPUTS, INPUTS, METADATA_LANES, 1, 1, top_k


class DirectDownReduce:
    def __init__(self, kernel, launch):
        self.kernel = kernel
        self.launch = launch
        self.geometry_cache = {}

    def __call__(self, *arguments):
        geometry = validate_geometry(arguments)
        device = arguments[0].device
        if device.type != "npu":
            raise ValueError("direct down-reduce requires NPU inputs")
        key = (device, geometry)
        buffers = self.geometry_cache.get(key)
        if buffers is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm up direct down-reduce geometry before capture")
            # Eight INT64 fields, padded to the binary's opParaSize=72.
            tiling = torch.tensor((*geometry, 0), dtype=torch.int64, device=device)
            workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            buffers = workspace, tiling
            self.geometry_cache[key] = buffers
        output = torch.empty((geometry[0] // geometry[-1], OUTPUTS), dtype=torch.float32, device=device)
        self.launch(self.kernel, [*arguments, output, *buffers], AI_CORE_BLOCKS)
        return output
