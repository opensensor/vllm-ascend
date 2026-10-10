# SPDX-License-Identifier: Apache-2.0
"""Disjoint uneven shared-expert channels; load-time placement only."""

import torch

from .weight_mapping import local_expert_range


def shared_expert_range(intermediate, tp_rank, tp_size):
    if any(type(n) is not int for n in (intermediate, tp_rank, tp_size)) or intermediate < tp_size:
        raise ValueError("uneven shared expert requires at least one trained channel per rank")
    return local_expert_range(intermediate, tp_size, tp_rank)


def place_uneven_shared_expert_tensor(params, name, tensor, intermediate, tp_rank, tp_size):
    first, stop = shared_expert_range(intermediate, tp_rank, tp_size)
    local = stop - first
    for projection in ("gate", "up", "down"):
        suffix = f".mlp.shared_expert.{projection}_proj.weight"
        if not name.endswith(suffix):
            continue
        target_name = name[: -len(suffix)] + (".mlp.shared_down" if projection == "down" else ".mlp.shared_gate_up")
        target = params.get(target_name)
        if target is None:
            return None
        if target.ndim != 2 or tensor.ndim != 2:
            raise ValueError("shared expert requires matrix weights")
        if projection == "down":
            expected = (target.shape[0], intermediate)
            target_shape = (target.shape[0], local)
            source = tensor[:, first:stop]
            destination = target
        else:
            expected = (intermediate, target.shape[1])
            target_shape = (2 * local, target.shape[1])
            source = tensor[first:stop]
            destination = target[:local] if projection == "gate" else target[local:]
        if tuple(tensor.shape) != expected or tuple(target.shape) != target_shape:
            raise ValueError("uneven shared expert checkpoint/destination shape mismatch")
        with torch.no_grad():
            destination.copy_(source.to(target.dtype))
        return target_name
    return None
