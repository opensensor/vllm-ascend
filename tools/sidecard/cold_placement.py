# SPDX-License-Identifier: Apache-2.0
"""Authenticated static expert ownership for the GPU-backbone runtime."""

import hashlib
import json
from pathlib import Path

from .layer_weights import GPU_LAYERS, NPU_LAYERS

EXPERTS = 288
PER_CHIP = EXPERTS // 2


def validate(hot):
    if not isinstance(hot, dict) or set(hot) != {str(layer) for layer in NPU_LAYERS}:
        raise ValueError("placement must cover every middle layer")
    for ids in hot.values():
        if (
            not isinstance(ids, list)
            or not ids
            or len(set(ids)) != len(ids)
            or any(type(expert) is not int or not 0 <= expert < EXPERTS for expert in ids)
            or ids != sorted(ids)
        ):
            raise ValueError("hot expert IDs must be unique sorted global IDs")
    return hot


def load(spec):
    path = Path(spec["expert_placement"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["expert_placement_sha256"]:
        raise ValueError("expert ownership failed authentication")
    record = json.loads(raw)
    if record["checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("expert ownership belongs to another checkpoint")
    return validate(record["gpu_hot_experts"])


def cuda_keys(hot, *, gate_layers=None):
    validate(hot)
    layers = sorted(set(GPU_LAYERS) | set(NPU_LAYERS)) if gate_layers is None else gate_layers
    return tuple(
        (layer, expert)
        for layer in layers
        if layer >= 3
        for expert in (hot[str(layer)] if layer in NPU_LAYERS else range(EXPERTS))
    )


def lookup(ids):
    """Map arbitrary canonical global IDs to sorted resident-bank offsets."""
    result = [-1] * EXPERTS
    for local, expert in enumerate(ids):
        result[expert] = local
    return result
