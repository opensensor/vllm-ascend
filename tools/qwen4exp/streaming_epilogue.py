# SPDX-License-Identifier: Apache-2.0
"""Builtin-FP16 activation and bounded routed-output finalization candidate.

The projected gate/up GM boundary survives: qualified nonlinear arithmetic and
cross-core column ownership cannot be silently replaced by an on-core fusion.
Callbacks retain the selected baseline and stream-order completion contract.
"""

from dataclasses import dataclass

import torch

from tools.qwen4exp.streaming_memory import BLOCKS, GROUP, LANES, MAX_K, N
from tools.qwen4exp.streaming_operands import MAX_EXPERTS, MAX_ROUTES, MAX_TOKENS, MAX_TOP_K

PROJECTED_COLUMNS = 1280
HIDDEN_COLUMNS = PROJECTED_COLUMNS // 2
NEXT_TILE_COLUMNS = 160


@dataclass(frozen=True)
class ColumnWindow:
    first_tile: int
    tile_count: int
    tile_columns: int = N

    @property
    def first_column(self):
        return self.first_tile * self.tile_columns

    @property
    def columns(self):
        return self.tile_count * self.tile_columns


@dataclass(frozen=True)
class WindowPlan:
    output_columns: int = MAX_K
    tiles_per_window: int = BLOCKS
    activation_policy: str = "cann_builtin_fp16"
    finalizer_policy: str = "cann_v2"
    tile_columns: int = N

    def __post_init__(self):
        if (
            type(self.output_columns) is not int
            or type(self.tile_columns) is not int
            or self.tile_columns not in (N, NEXT_TILE_COLUMNS)
            or not self.tile_columns <= self.output_columns <= MAX_K
            or self.output_columns % self.tile_columns
            or type(self.tiles_per_window) is not int
            or not 1 <= self.tiles_per_window <= BLOCKS
            or self.activation_policy != "cann_builtin_fp16"
            or self.finalizer_policy != "cann_v2"
        ):
            raise ValueError("unsupported column-window or arithmetic policy")

    @property
    def windows(self):
        tiles = self.output_columns // self.tile_columns
        return tuple(
            ColumnWindow(first, min(self.tiles_per_window, tiles - first), self.tile_columns)
            for first in range(0, tiles, self.tiles_per_window)
        )


@dataclass(frozen=True)
class PipelineResult:
    output: object
    windows: tuple
    boundary_epochs: tuple
    logical_buffer_bounds: tuple
    backend_scratch_measured: bool = False


def epilogue_buffer_bounds(rows, tokens, plan):
    """Logical exclusive tensor storage; allocator/weights/state are elsewhere.

    Conservatively retain projected input throughout, even after activation.
    One routed window plus finalizer casts/output coexist with the full output.
    Unknown builtin/CANN backend scratch must be measured separately at admission.
    Rows include peer capacity; active values are never inspected by Python.
    """
    if (
        type(rows) is not int
        or type(tokens) is not int
        or not 0 <= rows <= MAX_ROUTES
        or not 0 <= tokens <= MAX_TOKENS
        or not tokens <= rows <= tokens * MAX_TOP_K
    ):
        raise ValueError("invalid epilogue storage geometry")
    width = min(plan.tiles_per_window * plan.tile_columns, plan.output_columns)
    values = (
        ("projected_gate_up_fp16", rows * PROJECTED_COLUMNS * 2),
        ("builtin_activation_fp16", rows * HIDDEN_COLUMNS * 2),
        ("packed_hidden_limbs", rows * HIDDEN_COLUMNS),
        ("packed_hidden_scale_sum", rows * (HIDDEN_COLUMNS // GROUP) * LANES * 2 * 4),
        ("routed_window_fp16", rows * width * 2),
        ("finalized_window_fp32", tokens * width * 4),
        ("finalizer_output_fp16", tokens * width * 2),
        ("finalizer_route_scales_fp16", rows * 2),
        ("finalizer_inverse_int32", rows * 4),
        ("complete_output_fp32", tokens * plan.output_columns * 4),
    )
    return values


def _require_tensor(value, shape, dtype, device, label):
    if tuple(value.shape) != shape or value.dtype != dtype or value.device != device or not value.is_contiguous():
        raise ValueError(f"{label} violated shape/dtype/device/contiguity contract")


def run_streaming_epilogue(
    projected, dispatch, weights, group_ends, down_bank, *, activation, pack, columns, finalize, complete, plan=None
):
    """Run the complete activation→packed handoff→down-window→finalizer path.

    ``complete(stage, tensor)`` establishes the selected stream dependency before
    the consumer is submitted or storage is released. It must use same-stream
    ordering/device events; no host synchronization or device value read is
    required here. Completion must raise on failure. No result is exposed after
    a partial failure. The finalizer receives original weights and dispatch
    unchanged, preserving its baseline FP16 scale/output boundaries and order.
    """
    plan = WindowPlan() if plan is None else plan
    if not isinstance(plan, WindowPlan):
        raise ValueError("a validated WindowPlan is required")
    if (
        projected.ndim != 2
        or projected.shape[1] != PROJECTED_COLUMNS
        or projected.dtype != torch.float16
        or not projected.is_contiguous()
        or weights.ndim != 2
        or not 0 <= weights.shape[0] <= MAX_TOKENS
        or not 1 <= weights.shape[1] <= MAX_TOP_K
        or projected.shape[0] != weights.numel()
        or weights.device != projected.device
        or group_ends.ndim != 1
        or not 1 <= group_ends.numel() <= MAX_EXPERTS
        or group_ends.dtype != torch.int64
        or group_ends.device != projected.device
        or not group_ends.is_contiguous()
        or not all(callable(fn) for fn in (activation, pack, columns, finalize, complete))
    ):
        raise ValueError("unsupported epilogue geometry/dtype/callback")
    owner = getattr(columns, "__self__", None)
    if owner is not None and getattr(owner, "tile_columns", plan.tile_columns) != plan.tile_columns:
        raise ValueError("column window plan does not match native resource tile width")
    rows, tokens = projected.shape[0], weights.shape[0]
    bounds = epilogue_buffer_bounds(rows, tokens, plan)
    output = torch.empty((tokens, plan.output_columns), dtype=torch.float32, device=projected.device)
    if rows == 0:
        return PipelineResult(output, (), (), bounds)
    if dispatch.inverse_order.numel() != rows or dispatch.inverse_order.device != projected.device:
        raise ValueError("dispatch inverse order does not cover routed rows")
    epochs = []

    def ready(stage, tensor):
        complete(stage, tensor)
        epochs.append(stage)

    ready("projected_gate_up", projected)
    hidden = activation(projected)
    _require_tensor(hidden, (rows, HIDDEN_COLUMNS), torch.float16, projected.device, "builtin activation")
    ready("builtin_activation", hidden)
    prepared = tuple(pack(hidden, group_ends))
    shapes = ((rows, HIDDEN_COLUMNS // 2),) * 2 + ((rows, HIDDEN_COLUMNS // GROUP, LANES),) * 2
    dtypes = (torch.int8, torch.int8, torch.float32, torch.float32)
    if len(prepared) != 4:
        raise ValueError("hidden quantizer must return four packed operands")
    for value, shape, dtype in zip(prepared, shapes, dtypes):
        _require_tensor(value, shape, dtype, projected.device, "packed hidden")
        ready("packed_hidden", value)
    del hidden
    for index, window in enumerate(plan.windows):
        routed = columns(down_bank, prepared, group_ends, window.first_tile, window.tile_count)
        _require_tensor(routed, (rows, window.columns), torch.float16, projected.device, "down window")
        ready(f"routed_window_{index}", routed)
        combined = finalize(routed, dispatch, weights)
        _require_tensor(combined, (tokens, window.columns), torch.float32, projected.device, "finalized window")
        ready(f"finalized_window_{index}", combined)
        output[:, window.first_column : window.first_column + window.columns].copy_(combined)
        ready(f"stored_window_{index}", output)
        # Drop prior locals before submitting the next window; two routed
        # windows must not coexist during evaluation of a new callback result.
        del routed, combined
    ready("complete_output", output)
    return PipelineResult(output, plan.windows, tuple(epochs), bounds)


def epilogue_logical_cost(rows, tokens, plan=None):
    """Worst-case payload with every physical route local; not bus measurements.

    Windowing bounds allocation but does not eliminate projected/down traffic.
    More finalizer invocations repeat small casts and add the full output copy.
    Native/backend scratch, cache effects and task duration remain unmeasured.
    """
    plan = WindowPlan() if plan is None else plan
    bounds = dict(epilogue_buffer_bounds(rows, tokens, plan))
    calls = len(plan.windows) if rows else 0
    return {
        "logical_buffer_bounds": bounds,
        "full_routed_output_fp16_bytes": rows * plan.output_columns * 2,
        "routed_down_write_bytes": rows * plan.output_columns * 2,
        "routed_finalizer_read_bytes": rows * plan.output_columns * 2,
        "combined_window_store_read_write_bytes": tokens * plan.output_columns * 4 * 2,
        "finalizer_cast_write_bytes": calls * rows * 6,
        "builtin_activation_calls": int(rows > 0),
        "quantizer_calls": int(rows > 0),
        "projection_calls": calls,
        "finalizer_calls": calls,
        "backend_scratch_measured": False,
        "measured_bus_bytes": False,
        "hardware_validated": False,
        "predicted_speed_multiplier": None,
    }
