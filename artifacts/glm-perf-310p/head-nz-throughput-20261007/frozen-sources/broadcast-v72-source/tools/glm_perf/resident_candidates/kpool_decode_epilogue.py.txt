# SPDX-License-Identifier: Apache-2.0
"""Batch decode index bookkeeping, retaining each original [1, capacity] topk."""

import torch


def select_and_expand(logits, positions, topk_tokens, pool_size):
    if (
        logits.ndim != 2
        or positions.shape != (logits.shape[0],)
        or pool_size <= 1
        or topk_tokens <= 0
        or topk_tokens % pool_size
    ):
        raise ValueError("invalid pooled selection shapes or budget")
    rows, capacity = logits.shape
    budget = topk_tokens // pool_size
    count = min(budget, capacity)
    lengths = positions.to(torch.int32) + 1
    completed = lengths // pool_size
    selected = torch.full((rows, budget), -1, dtype=torch.int32, device=logits.device)
    if count == capacity:
        indices = torch.arange(count, dtype=torch.int32, device=logits.device).expand(rows, -1)
        valid = indices < completed[:, None]
    elif count and rows:
        indices = torch.cat([torch.topk(logits[row : row + 1], count, dim=-1).indices for row in range(rows)]).int()
        ranks = torch.arange(count, device=logits.device)
        # Rank masking is essential: 310P topk can alias valid indices for -inf padding.
        valid = (ranks < completed[:, None]) & (indices >= 0) & (indices < completed[:, None])
    else:
        indices = selected[:, :0]
        valid = indices >= 0
    selected[:, :count] = torch.where(valid, indices, -1)
    offsets = torch.arange(pool_size, dtype=torch.int32, device=logits.device)
    expanded = selected[:, :, None] * pool_size + offsets
    expanded.masked_fill_(selected[:, :, None] < 0, -1)
    tail_start = completed * pool_size
    tail_offsets = offsets[:-1]
    tail = tail_start[:, None] + tail_offsets
    tail.masked_fill_(tail_offsets >= (lengths - tail_start)[:, None], -1)
    return torch.cat((expanded.flatten(1), tail), dim=1)


def replacements(native_resources=None):
    from vllm_ascend.models.glm5next import sparse_attn_indexer_kpool as module

    original = module.SparseAttnIndexerKpool._select_tokens_fixed

    def select(self, queries, weights, positions, pool_size, indexer_metadata):
        cache = module._cache_tensor(self.k_cache)
        table = indexer_metadata.block_table
        pools = min((self.max_model_len + pool_size - 1) // pool_size, table.shape[1] * cache.shape[1])
        if not getattr(self, "live_kpool_score", False) or not module.supports_live_kpool_score(
            queries, cache, pool_size, pools
        ):
            return original(self, queries, weights, positions, pool_size, indexer_metadata)
        logits = module.score_kpool_paged(
            queries, weights, cache, table, indexer_metadata.cum_query_lens, positions, pools
        )
        expanded = select_and_expand(logits, positions, self.topk_tokens, pool_size)
        output = self.topk_indices_buffer[: queries.shape[0]]
        if output.shape[1] > expanded.shape[1]:
            output.fill_(-1)
        output[:, : expanded.shape[1]].copy_(expanded)
        return self.topk_indices_buffer

    return {"vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._select_tokens_fixed": select}
