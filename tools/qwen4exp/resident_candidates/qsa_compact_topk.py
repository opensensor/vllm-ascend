# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in resident backport of compact NPU indices in the existing fast top-k.

Only index representation changes. Score width, top-k options, selected set,
stable tie ordering and score arithmetic retain the qualified implementation.
There is no row tiling, selection policy change, device initialization or patch
on import. CPU and oversized-index inputs retain INT64 gather indices.
"""

import torch


def compact_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("k must be in [1, scores.shape[1]]")
    selected = torch.topk(scores, k, dim=1, sorted=False).indices
    compact_indices = scores.device.type == "npu" and scores.shape[1] <= torch.iinfo(torch.int32).max
    if compact_indices:
        selected = selected.to(torch.int32)
    index_order = torch.argsort(selected.to(torch.float32), dim=1, stable=True)
    if compact_indices:
        index_order = index_order.to(torch.int32)
    selected = selected.gather(1, index_order)
    selected_values = scores.gather(1, selected)
    value_order = torch.argsort(selected_values, dim=1, descending=True, stable=True)
    if compact_indices:
        value_order = value_order.to(torch.int32)
    return selected.gather(1, value_order)


def replacements():
    """Explicit installation only; the qualified stable decode path is intact."""
    return {"vllm_ascend.models.qwen4_exp.ops.qsa_indexer:_fast_topk_indices": compact_topk_indices}
