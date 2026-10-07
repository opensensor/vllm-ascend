# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in experiment; never loads a native library into a worker."""

from tools.glm_perf.kpool_prefill import select_prefill_request, supports_prefill_score
from vllm_ascend.models.glm5next.kpool_ops import dense_kpool_token_indices
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool, _cache_tensor

BASELINE_SELECT = SparseAttnIndexerKpool._select_tokens


def select_tokens(self, queries, weights, positions, pool_size, indexer_metadata):
    cache = _cache_tensor(self.k_cache)
    if not supports_prefill_score(queries, weights, cache, pool_size):
        return BASELINE_SELECT(self, queries, weights, positions, pool_size, indexer_metadata)
    num_tokens = indexer_metadata.num_actual_tokens
    if num_tokens > min(queries.shape[0], weights.shape[0], positions.shape[0]):
        raise RuntimeError("GLM kpool metadata exceeds the selection tensors")
    boundaries = indexer_metadata.cum_query_lens_cpu
    if boundaries is None:
        raise RuntimeError("GLM kpool needs host query boundaries from the scheduler")
    pool_lengths = indexer_metadata.seq_lens_cpu.tolist()
    if pool_lengths and max(pool_lengths) > indexer_metadata.block_table.shape[1] * cache.shape[1]:
        raise RuntimeError("GLM kpool block table is shorter than the request's pooled keys")
    self.topk_indices_buffer[:num_tokens].fill_(-1)
    query_ends = boundaries.tolist()
    for request, (start, end) in enumerate(zip(query_ends[:-1], query_ends[1:])):
        if start == end:
            continue
        num_pools = int(pool_lengths[request])
        if num_pools <= self.topk_tokens // pool_size:
            expanded = dense_kpool_token_indices(positions[start:end], self.topk_tokens, pool_size)
        else:
            expanded = select_prefill_request(
                queries[start:end],
                weights[start:end],
                cache,
                indexer_metadata.block_table[request : request + 1],
                positions[start:end],
                num_pools,
                self.topk_tokens,
                pool_size,
            )
        self.topk_indices_buffer[start:end, : expanded.shape[1]].copy_(expanded)
    return self.topk_indices_buffer


def replacements():
    return {
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._select_tokens": select_tokens
    }
