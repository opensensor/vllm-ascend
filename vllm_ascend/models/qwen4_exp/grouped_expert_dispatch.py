# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident routing descriptor for the Qwen4Exp expert bank.

The descriptor is deliberately independent of the expert GEMM backend. Its
group boundaries and permutations can be consumed by a future grouped W8A16
AI Core kernel without reading expert counts or selected ids on the host. The
current 310P fallback still reads the count vector to dispatch its supported
single-expert weight-only GEMMs.
"""

from dataclasses import dataclass

import torch

_MAX_EXACT_FLOAT32_INTEGER = 1 << 24


@dataclass(frozen=True)
class GroupedExpertDispatch:
    """Local expert routing, with peer-owned routes sorted into a final bin.

    ``order`` maps sorted routes to original (token, top-k slot) positions;
    ``inverse_order`` reverses it. ``counts`` has exactly one element per local
    expert and excludes the peer-owned sentinel. All tensors stay on the input
    device. A grouped kernel can use ``group_list`` as cumulative end offsets
    and ignore rows after the last offset.
    """

    order: torch.Tensor
    inverse_order: torch.Tensor
    token_indices: torch.Tensor
    route_weights: torch.Tensor
    counts: torch.Tensor

    @property
    def group_list(self) -> torch.Tensor:
        """Device-side cumulative expert group ends, including empty groups."""
        return self.counts.cumsum(dim=0)


def build_grouped_expert_dispatch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_local_experts: int,
    expert_offset: int,
    weight_dtype: torch.dtype,
) -> GroupedExpertDispatch:
    """Sort local expert routes without synchronizing with the host.

    This accepts unequal contiguous expert shards (for example 512 experts
    distributed as 86/86/85/85/85/85 over six ranks). Global top-k ids never
    change: peer-owned routes are placed after all local groups and can be
    mapped to a zero row when reconstructing this rank's partial output.
    """
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_ids and topk_weights must have the same [tokens, top_k] shape")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must contain int32 or int64 expert ids")
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk_ids and topk_weights must be on the same device")
    if num_local_experts <= 0 or expert_offset < 0:
        raise ValueError("num_local_experts must be positive and expert_offset nonnegative")

    num_tokens, top_k = topk_ids.shape
    num_routes = num_tokens * top_k
    pair_expert = topk_ids.reshape(-1) - expert_offset
    in_local = (pair_expert >= 0) & (pair_expert < num_local_experts)
    pair_expert = torch.where(in_local, pair_expert, num_local_experts)
    token_indices = torch.arange(num_tokens, device=topk_ids.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    route_weights = topk_weights.to(weight_dtype).reshape(-1, 1)

    # 310P sorts float32 keys on AI Core. The conversion is exact for the
    # bounded expert id and permutation ranges used by this model.
    expert_sort_key = pair_expert.to(torch.float32) if num_local_experts <= _MAX_EXACT_FLOAT32_INTEGER else pair_expert
    order = torch.argsort(expert_sort_key, stable=True)
    local_expert_ids = torch.arange(num_local_experts, device=topk_ids.device, dtype=pair_expert.dtype)
    counts = (pair_expert.unsqueeze(1) == local_expert_ids).sum(dim=0)
    inverse_sort_key = order.to(torch.float32) if num_routes <= _MAX_EXACT_FLOAT32_INTEGER else order
    inverse_order = torch.argsort(inverse_sort_key)
    return GroupedExpertDispatch(order, inverse_order, token_indices, route_weights, counts)
