# SPDX-License-Identifier: Apache-2.0
"""Unqualified tiled prefill scorer; importing this module changes no runtime.

The separately named native operator must be loaded before a resident trial.
Keep selection shapes equal to the baseline: native row/column padding is
removed before top-k, including exact ties and its 310P padding checks.
"""

import torch

from vllm_ascend.models.glm5next import kpool_ops

QUERY_TILE = 4
MAX_NATIVE_ROWS = 128
SCORE_ALIGNMENT = 8
HEADS = 32
DIM = 128


def supports_prefill_score(queries, weights, cache, pool_size):
    return (
        pool_size == 4
        and queries.ndim == 3
        and queries.shape[1:] == (HEADS, DIM)
        and weights.shape == queries.shape[:2]
        and cache.ndim == 4
        and cache.shape[-2:] == (1, DIM)
        and cache.dtype == torch.float16
        and cache.shape[1] > 0
        and cache.shape[1] % SCORE_ALIGNMENT == 0
        and cache.stride(-1) == 1
        and cache.stride(1) >= DIM
        and cache.stride(1) % 16 == 0
        and cache.stride(1) // 16 <= 65535
        and cache.stride(0) >= cache.shape[1] * cache.stride(1)
        and cache.stride(0) % 16 == 0
        and cache.storage_offset() % 16 == 0
    )


def score_prefill_paged(queries, weights, cache, table, positions, num_pools, *, native_op=None):
    """Score one request directly from shared pages, bounded to 128 rows/call."""
    if not supports_prefill_score(queries, weights, cache, 4):
        raise ValueError("unsupported tiled prefill score layout")
    rows = queries.shape[0]
    if not 0 < rows <= MAX_NATIVE_ROWS or positions.shape != (rows,):
        raise ValueError("prefill scorer needs 1..128 query positions")
    padded_pools = (num_pools + SCORE_ALIGNMENT - 1) // SCORE_ALIGNMENT * SCORE_ALIGNMENT
    if table.ndim != 2 or table.shape[0] != 1 or not 0 < padded_pools <= table.shape[1] * cache.shape[1]:
        raise ValueError("prefill scorer needs one complete request page table")
    padded_rows = (rows + QUERY_TILE - 1) // QUERY_TILE * QUERY_TILE
    rotated = kpool_ops.hadamard128(queries).bfloat16().half()
    if padded_rows == rows:
        padded_query = rotated.contiguous()
        padded_weights = weights.float().contiguous()
        padded_positions = positions.to(torch.int32).contiguous()
    else:
        padded_query = torch.zeros((padded_rows, HEADS, DIM), dtype=torch.float16, device=queries.device)
        padded_weights = torch.zeros((padded_rows, HEADS), dtype=torch.float32, device=queries.device)
        padded_positions = torch.full((padded_rows,), -1, dtype=torch.int32, device=queries.device)
        padded_query[:rows].copy_(rotated)
        padded_weights[:rows].copy_(weights)
        padded_positions[:rows].copy_(positions)
    elements = cache.untyped_storage().nbytes() // cache.element_size()
    flat = cache.as_strided((elements,), (1,), storage_offset=0)
    op = native_op if native_op is not None else torch.ops._C_ascend.npu_glm_kpool_prefill_score_310
    scores = op(
        padded_query,
        padded_weights,
        flat,
        table.to(torch.int32).contiguous(),
        torch.full((1,), padded_rows, dtype=torch.int32, device=queries.device),
        padded_positions,
        padded_pools,
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        cache.storage_offset(),
    )
    return scores[:rows, :num_pools]


def select_prefill_request(
    queries, weights, cache, table, positions, num_pools, topk_tokens, pool_size, *, native_op=None
):
    """Preserve baseline selection geometry while eliminating head score RAM."""
    if queries.shape[0] == 0:
        return torch.empty((0, topk_tokens + pool_size - 1), dtype=torch.int32, device=queries.device)
    if pool_size != 4:
        raise ValueError("native prefill scoring requires four-token pools")
    query_chunk_size = max(1, kpool_ops.MAX_KPOOL_SCORE_ELEMENTS // max(1, queries.shape[1] * num_pools))
    expanded_chunks = []
    for start in range(0, queries.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, queries.shape[0])
        score_chunks = []
        for first in range(start, end, MAX_NATIVE_ROWS):
            last = min(first + MAX_NATIVE_ROWS, end)
            score_chunks.append(
                score_prefill_paged(
                    queries[first:last],
                    weights[first:last],
                    cache,
                    table,
                    positions[first:last],
                    num_pools,
                    native_op=native_op,
                )
            )
        scores = score_chunks[0] if len(score_chunks) == 1 else torch.cat(score_chunks)
        selected, _, tail_starts, tail_counts = kpool_ops.select_kpool_groups(
            scores, positions[start:end], topk_tokens, pool_size, scores_are_causal=True
        )
        expanded_chunks.append(kpool_ops.expand_kpool_groups(selected, tail_starts, tail_counts, pool_size))
        del scores, score_chunks
    return expanded_chunks[0] if len(expanded_chunks) == 1 else torch.cat(expanded_chunks)
