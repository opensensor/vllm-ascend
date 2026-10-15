# SPDX-License-Identifier: Apache-2.0
"""Versioned direct gate/up resources for isolated and resident experiments.

This adapter deliberately supports only Qwen's INT32 routed decode geometry.
It never changes OPP lookup, registers code inside capture, or replaces weights.
"""

import torch

GATE_UP_COLUMNS = 1280
GATE_UP_INPUTS = 2560
GROUP_SIZE = 128
METADATA_LANES = 8
MAX_DECODE_ROUTES = 128
WORKSPACE_BYTES = 16 * 1024 * 1024
AI_CORE_BLOCKS = 8


def validate_geometry(arguments):
    """Check metadata without reading device values or synchronizing."""
    if len(arguments) != 9:
        raise ValueError("direct gate/up requires nine inputs")
    low, high, xs, sums, codes, scale, offset, weight_sum, routes = arguments
    expected_dtypes = (
        torch.int8,
        torch.int8,
        torch.float32,
        torch.float32,
        torch.int8,
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
    )
    if any(t.dtype != dtype for t, dtype in zip(arguments, expected_dtypes)):
        raise ValueError("direct gate/up input dtypes do not match the INT32 decode binary")
    if any(t.device != low.device or not t.is_contiguous() for t in arguments):
        raise ValueError("direct gate/up requires contiguous inputs on one device")
    if low.ndim != 2 or codes.ndim != 3 or routes.ndim != 1:
        raise ValueError("invalid direct gate/up input ranks")
    rows, packed_width = low.shape
    experts, columns, weight_width = codes.shape
    output_rows = routes.numel()
    if (
        rows <= 0
        or output_rows < rows
        or output_rows > MAX_DECODE_ROUTES
        or output_rows % rows
        or packed_width != GATE_UP_INPUTS // 2
        or columns != GATE_UP_COLUMNS
        or weight_width != packed_width
        or experts <= 0
    ):
        raise ValueError("unsupported direct gate/up decode geometry")
    activation_metadata = (rows, GATE_UP_INPUTS // GROUP_SIZE, METADATA_LANES)
    weight_metadata = (experts, columns, GATE_UP_INPUTS // GROUP_SIZE)
    if (
        high.shape != low.shape
        or xs.shape != activation_metadata
        or sums.shape != activation_metadata
        or any(t.shape != weight_metadata for t in (scale, offset, weight_sum))
    ):
        raise ValueError("direct gate/up metadata shape mismatch")
    return output_rows, experts, columns, GATE_UP_INPUTS, METADATA_LANES, 1, output_rows // rows


class DirectGateUp:
    """Own a kernel handle and small geometry caches until worker exit.

    The caller supplies registered bridge operations; construction does not
    load a Python extension. NativeSession.prepare() owns binary registration.
    Warmup creates tiling/workspace buffers before affected graphs are captured.
    """

    def __init__(self, kernel, launch):
        self.kernel = kernel
        self.launch = launch
        self.geometry_cache = {}

    def __call__(self, *arguments):
        geometry = validate_geometry(arguments)
        device = arguments[0].device
        if device.type != "npu":
            raise ValueError("direct gate/up requires NPU inputs")
        key = (device, geometry)
        buffers = self.geometry_cache.get(key)
        if buffers is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm up direct gate/up geometry before graph capture")
            # The CANN struct occupies seven INT64 fields. Pad to the binary's
            # opParaSize=64; workspace is unused by the source, reserved here.
            tiling = torch.tensor((*geometry, 0), dtype=torch.int64, device=device)
            workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            buffers = workspace, tiling
            self.geometry_cache[key] = buffers
        output = torch.empty((geometry[0], geometry[2]), dtype=torch.float16, device=device)
        self.launch(self.kernel, [*arguments, output, *buffers], AI_CORE_BLOCKS)
        return output
