# SPDX-License-Identifier: Apache-2.0
"""Bounded grouped token operands; callbacks never return host route decisions.

The gather callback must obey device group ends, leave peer rows unread, and
preserve dispatch order. This orchestration does not replace sparse decode or
W8A16 MTP dispatch. All byte estimates are logical payloads, not bus traffic.
"""

from dataclasses import dataclass

import torch

from tools.qwen4exp.streaming_memory import GROUP, LANES, MAX_K

MAX_TOKENS = 2560
MAX_TOP_K = 10
MAX_ROUTES = MAX_TOKENS * MAX_TOP_K
MAX_EXPERTS = 128


@dataclass(frozen=True)
class GroupedOperands:
    token_operands: tuple
    local_operands: tuple
    dispatch: object
    sorted_tokens: object
    group_ends: object
    physical_rows: int


def prepare_grouped_operands(inputs, weights, ids, *, pack, dispatch, gather, num_local_experts, expert_offset=0):
    """Pack each token once, then compact its local routes with fixed capacity.

    Shapes and dtypes are host metadata. Neither route IDs nor group ends are
    read by Python. The injected dispatcher owns stable sorting, renormalized
    route weights, and int64 device boundaries. The injected native gather owns
    bounds checks and only writes the active local prefix. Unwritten peer rows
    are deliberately undefined and projections must skip/zero them via ends.
    """
    if (
        inputs.ndim != 2
        or not 0 <= inputs.shape[0] <= MAX_TOKENS
        or not GROUP <= inputs.shape[1] <= MAX_K
        or inputs.shape[1] % GROUP
        or weights.ndim != 2
        or ids.shape != weights.shape
        or weights.shape[0] != inputs.shape[0]
        or not 1 <= weights.shape[1] <= MAX_TOP_K
        or type(num_local_experts) is not int
        or not 1 <= num_local_experts <= MAX_EXPERTS
        or type(expert_offset) is not int
        or expert_offset < 0
        or weights.device != inputs.device
        or ids.device != inputs.device
    ):
        raise ValueError("unsupported grouped operand geometry")
    physical_rows = inputs.shape[0] * weights.shape[1]
    if not physical_rows:
        return GroupedOperands((), (), None, None, None, 0)
    routed = dispatch(weights, ids, num_local_experts=num_local_experts, expert_offset=expert_offset)
    order = routed.order
    if (
        order.ndim != 1
        or order.numel() != physical_rows
        or routed.token_indices.shape != order.shape
        or routed.group_list.shape != (num_local_experts,)
        or routed.group_list.dtype != torch.int64
        or any(value.device != inputs.device for value in (order, routed.token_indices, routed.group_list))
    ):
        raise ValueError("dispatcher violated fixed route capacity")
    sorted_tokens = routed.token_indices.index_select(0, order).to(torch.int32).contiguous()
    group_ends = routed.group_list.contiguous()
    token_operands = tuple(pack(inputs))
    groups = inputs.shape[1] // GROUP
    expected = ((inputs.shape[0], inputs.shape[1] // 2),) * 2 + ((inputs.shape[0], groups, LANES),) * 2
    if len(token_operands) != 4 or any(
        tuple(value.shape) != shape or value.device != inputs.device or not value.is_contiguous()
        for value, shape in zip(token_operands, expected)
    ):
        raise ValueError("quantizer violated packed operand ABI")
    if tuple(value.dtype for value in token_operands) != (torch.int8, torch.int8, torch.float32, torch.float32):
        raise ValueError("quantizer violated packed operand dtype")
    local_operands = tuple(gather(token_operands, sorted_tokens, group_ends))
    if len(local_operands) != 4 or any(
        tuple(value.shape) != (physical_rows, *original.shape[1:])
        or value.dtype != original.dtype
        or value.device != original.device
        or not value.is_contiguous()
        for value, original in zip(local_operands, token_operands)
    ):
        raise ValueError("gather violated fixed local operand ABI")
    return GroupedOperands(token_operands, local_operands, routed, sorted_tokens, group_ends, physical_rows)


def operand_transfer_cost(tokens, top_k, local_rows, width, output_tiles):
    """Compare complete operand payloads including repeated projection reads.

    Compact gather reads+writes each live route once. Both layouts subsequently
    read each route for every expert output tile. Indexed reads may repeat token
    addresses; no cache or coalescing benefit is assumed. Fixed allocations are
    reported separately from executed traffic. This model does not choose a
    backend without hardware projection/dispatch timings.
    """
    values = (tokens, top_k, local_rows, width, output_tiles)
    if any(type(value) is not int for value in values):
        raise ValueError("cost geometry must use host integer metadata")
    if (
        not 0 <= tokens <= MAX_TOKENS
        or not 1 <= top_k <= MAX_TOP_K
        or not 0 <= local_rows <= tokens * top_k
        or not GROUP <= width <= MAX_K
        or width % GROUP
        or output_tiles < 1
    ):
        raise ValueError("unsupported cost geometry")
    row_bytes = width + 2 * (width // GROUP) * LANES * 4
    projection = local_rows * row_bytes * output_tiles
    gather = local_rows * row_bytes * 2
    return {
        "packed_bytes_per_row": row_bytes,
        "token_operand_storage_bytes": tokens * row_bytes,
        "compact_operand_storage_bytes": tokens * top_k * row_bytes,
        "indexed_projection_read_bytes": projection,
        "compact_gather_read_write_bytes": gather,
        "compact_projection_read_bytes": projection,
        "indexed_total_operand_bytes": projection,
        "compact_total_operand_bytes": gather + projection,
        "common_quantizer_output_write_bytes": tokens * row_bytes,
        "indexed_operand_bytes_with_quantizer_write": tokens * row_bytes + projection,
        "compact_operand_bytes_with_quantizer_write": tokens * row_bytes + gather + projection,
        "quantized_tokens": tokens,
        "selected_backend": None,
        "measured_bus_bytes": False,
        "hardware_validated": False,
    }
