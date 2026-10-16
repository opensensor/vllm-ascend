# SPDX-License-Identifier: Apache-2.0
"""Portable contract for the qualified expert-major deterministic reducer."""

import math

TOP_K = 8
WIDTH = 4096
INTERMEDIATE = 2048
EXPERTS = 144
NATIVE_LAYOUT = "cube_n128_k256_v1"
SCALE_LAYOUT = "fp16_storage_fp32_compute_v1"


def route_order(ids, weights, rank):
    """Return live original slots in stable expert-major order, retaining repeats.

    This is the frozen padded reducer's addition order, not original top-k order.
    Native metadata computes this on device; this CPU function is a test oracle.
    """
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    if len(ids) != TOP_K or len(weights) != TOP_K:
        raise ValueError("decode requires exactly eight route slots")
    if any(type(value) is not int or not 0 <= value < 2 * EXPERTS for value in ids):
        raise ValueError("invalid global expert ID")
    if any(isinstance(value, bool) or not math.isfinite(value) for value in weights):
        raise ValueError("route weights must be finite")
    return tuple(
        sorted(
            (
                slot
                for slot, expert in enumerate(ids)
                if rank * EXPERTS <= expert < (rank + 1) * EXPERTS and weights[slot]
            ),
            key=lambda slot: ids[slot],
        )
    )


def workspace_bytes():
    """Eager resident bank scratch; ignores allocator alignment and kernel UB.

    Prepared batch slots retain the frozen expert-index addressing, including
    unused slots. Reducing that layout is a separate kernel/ABI experiment.
    """
    slots = (TOP_K + 30) // 31 + EXPERTS
    return dict(
        input_low=WIDTH,
        input_high=1,
        input_scales=WIDTH // 32 * 4,
        routed_input=slots * (WIDTH // 64) * (2 * 32 * 32),
        routed_input_scales=slots * 32 * 128 * 4,
        hidden_low=slots * (INTERMEDIATE // 64) * (2 * 32 * 32),
        hidden_high=1,
        hidden_scales=slots * (INTERMEDIATE // 128) * 32 * 4 * 4,
        route_values=TOP_K * WIDTH * 2,
        output=WIDTH * 4,
        route_order=TOP_K * 8,
        route_ends=EXPERTS * 8,
        route_ranks=TOP_K * 4,
    )


def route_cases(rank, layer=18):
    """Bounded ownership, ordering, duplicate, and zero-weight numerical cases."""
    if type(layer) is not int or not 18 <= layer <= 32:
        raise ValueError("unsupported resident middle layer")
    route_order([0] * TOP_K, [0.0] * TOP_K, rank)
    owner, peer = rank * EXPERTS, (1 - rank) * EXPERTS
    weights = [0.125, 0.75, 0.0, 1.375, 0.25, 0.875, 0.625, 0.5]
    cases = []
    for count in range(TOP_K + 1):
        ids = [owner + (layer * 13 + slot * 17) % EXPERTS if slot < count else peer + slot for slot in range(TOP_K)]
        cases.append(dict(name=f"owned-{count}", ids=ids, weights=[0.137 + slot * 0.013 for slot in range(TOP_K)]))
    for slot in range(TOP_K):
        ids = [peer + index for index in range(TOP_K)]
        ids[slot] = owner + (layer * 7 + slot * 19) % EXPERTS
        cases.append(dict(name=f"single-slot-{slot}", ids=ids, weights=[0.25 + slot * 0.07] * TOP_K))
    cases.extend(
        (
            dict(name="duplicate-eight", ids=[owner + 31] * TOP_K, weights=weights),
            dict(
                name="duplicate-peer-zero",
                ids=[owner, peer, owner, owner + 143, peer + 143, owner, owner + 143, peer],
                weights=weights,
            ),
            dict(
                name="reverse-order", ids=[owner + value for value in (143, 1, 31, 0, 31, 142, 2, 31)], weights=weights
            ),
            dict(name="zero-weights", ids=[owner + slot for slot in range(TOP_K)], weights=[0.0] * TOP_K),
        )
    )
    return cases


def ordered_reduce(route_values, ids, weights, rank):
    """Independent CPU arithmetic oracle: rounded FP16 rows, FP32 mul/add.

    Values correspond to original route slots. Peer and zero-weight rows can be
    NaN poison and must not be read. Device kernels store stable-expert rows;
    the frozen ranks permutation produces this same ordered accumulation.
    """
    import torch

    if route_values.ndim != 2 or route_values.shape[0] != TOP_K or route_values.device.type != "cpu":
        raise ValueError("reference requires eight CPU route rows")
    result = torch.zeros(route_values.shape[1], dtype=torch.float32)
    for slot in route_order(ids, weights, rank):
        result = result + route_values[slot].half().float() * torch.tensor(weights[slot], dtype=torch.float32)
    return result
