# SPDX-License-Identifier: Apache-2.0
"""Experimental decode-only rotation fusion; load rotation_v1 before applying."""

import torch

from vllm_ascend.models.glm5next.ops.kpool_native import score_kpool_paged as baseline_score


def replacements(native_resources):
    rotate = native_resources["rotation_v1"]

    def score(queries, weights, cache, table, boundaries, positions, num_pools):
        if (
            queries.dtype not in (torch.float16, torch.bfloat16)
            or queries.ndim != 3
            or queries.shape[1:] != (32, 128)
            or not 1 <= queries.shape[0] <= 8
            or not queries.is_contiguous()
        ):
            return baseline_score(queries, weights, cache, table, boundaries, positions, num_pools)
        elements = cache.untyped_storage().nbytes() // cache.element_size()
        flat = cache.as_strided((elements,), (1,), storage_offset=0)
        return torch.ops._C_ascend.npu_glm_kpool_score_310(
            rotate(queries),
            weights.float().contiguous(),
            flat,
            table.to(torch.int32).contiguous(),
            boundaries.to(torch.int32).contiguous(),
            positions.to(torch.int32).contiguous(),
            num_pools,
            cache.shape[0],
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            cache.storage_offset(),
        )

    return {"vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:score_kpool_paged": score}
