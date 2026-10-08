# SPDX-License-Identifier: Apache-2.0
import torch
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import _cache_tensor
from vllm_ascend.models.glm5next.ops.kpool_native import score_kpool_paged, supports_live_kpool_score
from vllm_ascend.models.glm5next.kpool_ops import score_kpool, select_kpool_groups, expand_kpool_groups

def _select_tokens_fixed(self, queries, weights, positions, pool_size, indexer_metadata):
    """Select decode rows with device-only lengths and fixed cache bounds.

    A captured row can cross the dense/sparse threshold or move to another
    request on replay. Read its current request and pages on device. Mask
    unused pages before scoring, including recycled pages containing NaNs.
    Prefill keeps the host-bounded batched path.
    """
    cache = _cache_tensor(self.k_cache)
    block_size = cache.shape[1]
    table = indexer_metadata.block_table
    num_pools = min((self.max_model_len + pool_size - 1) // pool_size, table.shape[1] * block_size)
    supported = supports_live_kpool_score(queries, cache, pool_size, num_pools)
    if not supported:
        raise RuntimeError(f"live scorer cannot handle query={queries.shape} cache={cache.shape} stride={cache.stride()} pools={num_pools}")
    if not getattr(self, "_live_score_logged", False):
        if torch.distributed.get_rank() == 0:
            print(f"GLM_LIVE_KPOOL selected=True rows={queries.shape[0]} pools={num_pools} cache_stride={cache.stride()}", flush=True)
        self._live_score_logged = True
    if True and supports_live_kpool_score(
        queries, cache, pool_size, num_pools
    ):
        logits = score_kpool_paged(
            queries, weights, cache, table, indexer_metadata.cum_query_lens, positions, num_pools
        )
        self.topk_indices_buffer[: queries.shape[0]].fill_(-1)
        # Preserve the existing per-row topk shapes and tie behavior.
        for row in range(queries.shape[0]):
            selected, _, tail_starts, tail_counts = select_kpool_groups(
                logits[row : row + 1], positions[row : row + 1], self.topk_tokens, pool_size
            )
            expanded = expand_kpool_groups(selected, tail_starts, tail_counts, pool_size)
            self.topk_indices_buffer[row : row + 1, : expanded.shape[1]].copy_(expanded)
        return self.topk_indices_buffer
    pool_ids = torch.arange(num_pools, device=cache.device, dtype=torch.long)
    row_ids = torch.arange(queries.shape[0], device=cache.device, dtype=indexer_metadata.cum_query_lens.dtype)
    requests = torch.searchsorted(indexer_metadata.cum_query_lens, row_ids, right=True)
    requests = requests.clamp(max=table.shape[0] - 1).long()
    self.topk_indices_buffer[: queries.shape[0]].fill_(-1)
    for row in range(queries.shape[0]):
        pages = table.index_select(0, requests[row : row + 1])[0]
        complete = (positions[row] + 1) // pool_size
        valid = pool_ids < complete
        physical_pages = pages[pool_ids // block_size].long()
        physical_pages = torch.where(valid, physical_pages, 0)
        keys = cache[physical_pages, pool_ids % block_size, 0]
        keys = torch.where(valid[:, None], keys, 0)
        logits = score_kpool(queries[row : row + 1], weights[row : row + 1], keys)
        selected, _, tail_starts, tail_counts = select_kpool_groups(
            logits, positions[row : row + 1], self.topk_tokens, pool_size
        )
        expanded = expand_kpool_groups(selected, tail_starts, tail_counts, pool_size)
        self.topk_indices_buffer[row : row + 1, : expanded.shape[1]].copy_(expanded)
    return self.topk_indices_buffer

def replacements():
    return {"vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._select_tokens_fixed": _select_tokens_fixed}
