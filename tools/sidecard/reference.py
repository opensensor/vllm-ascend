# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for a routed-expert boundary, not an accelerator transport.

Workers return unweighted expert outputs. The coordinator combines in original
top-k slot order, scales once, and adds the CUDA-owned shared output once.
"""

import math
import struct
from dataclasses import dataclass

from .placement import ExpertPlacement


def fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class Route:
    token: int
    slot: int
    expert: int


@dataclass(frozen=True)
class ExpertRequest:
    execution_id: str
    layer_id: int
    owner: str
    routes: tuple[Route, ...]
    token_ids: tuple[int, ...]
    hidden: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ExpertReply:
    execution_id: str
    layer_id: int
    owner: str
    routes: tuple[Route, ...]
    outputs: tuple[tuple[float, ...], ...]


def _validate_rows(rows, width):
    if any(len(row) != width or any(not math.isfinite(value) for value in row) for row in rows):
        raise ValueError("invalid row width or nonfinite value")


def dispatch(placement, execution_id, layer_id, hidden, expert_ids, weights):
    """Compact token rows per active owner; route weights remain with coordinator."""
    layer = placement.layer(layer_id)
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("execution identity must be nonempty")
    if not hidden or not hidden[0] or len(hidden) != len(expert_ids) or len(hidden) != len(weights):
        raise ValueError("hidden, expert ids, and weights require matching nonempty token rows")
    width = len(hidden[0])
    _validate_rows(hidden, width)
    top_k = len(expert_ids[0])
    if not 1 <= top_k <= len(layer):
        raise ValueError("invalid top-k")
    _validate_rows(weights, top_k)
    routes_by_owner = {owner: [] for owner in placement.owners}
    for token, (ids, row_weights) in enumerate(zip(expert_ids, weights)):
        if len(ids) != top_k or len(set(ids)) != top_k:
            raise ValueError("top-k expert ids must be unique and have consistent width")
        if any(weight < 0 for weight in row_weights):
            raise ValueError("route weights must be nonnegative")
        for slot, expert in enumerate(ids):
            if type(expert) is not int or not 0 <= expert < len(layer):
                raise ValueError("expert id out of range")
            routes_by_owner[layer[expert]].append(Route(token, slot, expert))
    requests = []
    for owner, routes in routes_by_owner.items():
        if not routes:
            continue
        token_ids = tuple(sorted({route.token for route in routes}))
        requests.append(
            ExpertRequest(
                execution_id, layer_id, owner, tuple(routes), token_ids, tuple(tuple(hidden[t]) for t in token_ids)
            )
        )
    return tuple(requests)


def validate_shared_output(hidden, shared, routed_scale):
    if len(shared) != len(hidden) or not math.isfinite(routed_scale) or routed_scale <= 0:
        raise ValueError("invalid shared output or routed scale")
    _validate_rows(shared, len(hidden[0]))


def validated_outputs(
    placement: ExpertPlacement,
    execution_id: str,
    layer_id: int,
    hidden,
    expert_ids,
    weights,
    replies,
    shared,
    routed_scale: float,
):
    """Check complete ownership/identity/shape coverage before any reduction."""
    requests = dispatch(placement, execution_id, layer_id, hidden, expert_ids, weights)
    width = len(hidden[0])
    validate_shared_output(hidden, shared, routed_scale)
    expected = {request.owner: set(request.routes) for request in requests}
    outputs = {}
    seen_owners = set()
    for reply in replies:
        if reply.execution_id != execution_id or reply.layer_id != layer_id:
            raise ValueError("stale execution or layer reply")
        if reply.owner not in expected or reply.owner in seen_owners:
            raise ValueError("unknown or duplicate owner reply")
        if len(reply.routes) != len(reply.outputs) or len(set(reply.routes)) != len(reply.routes):
            raise ValueError("invalid or duplicate reply routes")
        if set(reply.routes) != expected[reply.owner]:
            raise ValueError("missing, foreign, or misowned route output")
        _validate_rows(reply.outputs, width)
        outputs.update(zip(reply.routes, reply.outputs))
        seen_owners.add(reply.owner)
    if seen_owners != set(expected):
        raise ValueError("missing owner reply")
    return outputs


def combine(
    placement: ExpertPlacement,
    execution_id: str,
    layer_id: int,
    hidden,
    expert_ids,
    weights,
    replies,
    shared,
    routed_scale: float,
):
    """Validate complete replies and use FP32, non-FMA, original-slot reduction."""
    outputs = validated_outputs(
        placement, execution_id, layer_id, hidden, expert_ids, weights, replies, shared, routed_scale
    )
    width = len(hidden[0])
    result = []
    for token, ids in enumerate(expert_ids):
        accumulated = [0.0] * width
        for slot, expert in enumerate(ids):
            row = outputs[Route(token, slot, expert)]
            weight = fp32(weights[token][slot])
            accumulated = [fp32(total + fp32(weight * fp32(value))) for total, value in zip(accumulated, row)]
        result.append(
            tuple(
                fp32(fp32(total * fp32(routed_scale)) + fp32(value)) for total, value in zip(accumulated, shared[token])
            )
        )
    return tuple(result)
