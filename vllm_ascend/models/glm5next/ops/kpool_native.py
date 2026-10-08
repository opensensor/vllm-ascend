# SPDX-License-Identifier: Apache-2.0
"""Opt-in paged scoring for 310P GLM graph decode.

The cache remains a view of shared backing storage. Never make this entire
view contiguous: a long-context cache can span gigabytes across its pages.
"""

import torch

from vllm_ascend.models.glm5next.kpool_ops import hadamard128

KPOOL_HEADS = 32
KPOOL_DIM = 128
KPOOL_SIZE = 4
MAX_GRAPH_ROWS = 8
SCORE_ALIGNMENT = 8


def supports_live_kpool_score(queries, cache, pool_size, num_pools):
    return (
        pool_size == KPOOL_SIZE
        and queries.ndim == 3
        and 0 < queries.shape[0] <= MAX_GRAPH_ROWS
        and queries.shape[1:] == (KPOOL_HEADS, KPOOL_DIM)
        and cache.ndim == 4
        and cache.shape[-2:] == (1, KPOOL_DIM)
        and cache.dtype == torch.float16
        and cache.stride(-1) == 1
        and cache.stride(1) >= KPOOL_DIM
        and cache.stride(1) % 16 == 0
        and cache.stride(0) >= cache.shape[1] * cache.stride(1)
        and cache.stride(0) % 16 == 0
        and cache.storage_offset() % 16 == 0
        and num_pools > 0
        and num_pools % SCORE_ALIGNMENT == 0
    )


def score_kpool_paged(queries, weights, cache, table, boundaries, positions, num_pools):
    """Keep score shape fixed while the kernel reads current device lengths.

    Cumulative boundaries contain request ends, without an initial zero.
    Query rotation and BF16 rounding match score_kpool. The experimental Cube
    operand uses FP16 after that round; very small/out-of-range BF16 values
    are not exactly representable, so serving qualification is still required.
    """
    elements = cache.untyped_storage().nbytes() // cache.element_size()
    flat = cache.as_strided((elements,), (1,), storage_offset=0)
    rotated = hadamard128(queries).to(torch.bfloat16).to(torch.float16).contiguous()
    return torch.ops._C_ascend.npu_glm_kpool_score_310(
        rotated,
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
