# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative logical payload bound for grouped native INT4 routes.

This excludes backend-private GEMM workspaces, allocator fragmentation, weights
and the caller's persistent inputs/outputs. It is not a device-memory limit.
"""

_MIB = 1 << 20
_ROUTE_INDEX_BYTES = 32
_FP16_BYTES = 2
_FP32_BYTES = 4
_INT64_BYTES = 8
_PACKED_PLANES_BYTES_PER_ELEMENT = 4


def grouped_route_chunk_tokens(
    requested_tokens: int,
    *,
    top_k: int,
    hidden: int,
    intermediate: int,
    local_experts: int,
    scratch_mib: int,
    histogram_counts: bool,
) -> int:
    if type(scratch_mib) is not int or scratch_mib < 0:
        raise ValueError("grouped_route_scratch_mib must be a nonnegative integer")
    if min(requested_tokens, top_k, hidden, intermediate, local_experts) <= 0:
        raise ValueError("route workspace geometry must be positive")
    if scratch_mib == 0:
        return requested_tokens
    # Bound simultaneously live route operands rather than relying on Python
    # reference destruction to make the previous projection disappear:
    # FP16 expanded input, gate/up, activation, down; FP32 converted/down
    # weighting/unpermutation and torch SwiGLU intermediates; activation
    # pack's four integer planes.
    route_bytes = (
        _FP16_BYTES * (hidden + 3 * intermediate + hidden)
        + _FP32_BYTES * (3 * hidden + 4 * intermediate)
        + _PACKED_PLANES_BYTES_PER_ELEMENT * (hidden + intermediate)
        + _ROUTE_INDEX_BYTES
        + (0 if histogram_counts else local_experts)
    )
    # Histogram output is fixed-size; reserve both keys/counts buffers.
    fixed_bytes = (local_experts + 1) * 2 * _INT64_BYTES
    token_bytes = top_k * route_bytes + _FP16_BYTES * hidden
    capacity = (scratch_mib * _MIB - fixed_bytes) // token_bytes
    if capacity < 1:
        raise ValueError("grouped route scratch budget cannot hold one token")
    return min(requested_tokens, capacity)
