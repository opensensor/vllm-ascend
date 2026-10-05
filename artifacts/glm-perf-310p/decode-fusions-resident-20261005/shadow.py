# SPDX-License-Identifier: Apache-2.0
"""Experimental decode-only rotation fusion; load rotation_v1 before applying."""

import torch

from vllm_ascend.models.glm5next.ops.kpool_native import score_kpool_paged as baseline_score


def base_replacements(native_resources):
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


def replacements(native_resources):
    from tools.glm_perf.resident_worker import ResidentWorkerExtension
    from vllm_ascend.models.glm5next.kpool_ops import hadamard128

    native = native_resources["rotation_v1"]
    counters = {}

    def checked(query):
        # First invocation is graph warmup. Subsequent captures/replays share
        # the same counters; status reads happen only outside inference.
        rows = query.shape[0]
        if rows not in counters:
            counters[rows] = torch.zeros(2, dtype=torch.float32, device=query.device)
        actual = native(query)
        expected = hadamard128(query).bfloat16().half()
        counters[rows][0].add_((actual != expected).float().sum())
        counters[rows][1].add_(query.numel())
        return actual

    def status(self):
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["rotation_parity"] = {str(rows): value.cpu().tolist() for rows, value in counters.items()}
        return receipt

    patches = base_replacements({"rotation_v1": checked})
    patches["vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"] = status
    return patches
