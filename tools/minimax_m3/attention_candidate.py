# SPDX-License-Identifier: Apache-2.0
"""Experimental head-batched attention; original sparse block selection retained."""

import torch

from vllm_ascend._310p.ops.minimax_m3 import ATTENTION_QUERY_TILE


def paged_sparse_attention_batched_heads(
    query,
    key,
    value,
    index_query,
    index_key,
    key_cache,
    value_cache,
    index_cache,
    *,
    query_starts,
    seq_lens,
    block_table,
    index_block_table,
    slot_mapping,
    index_slot_mapping,
    max_seq_len,
    topk_blocks,
    scale,
    init_blocks=0,
    local_blocks=1,
):
    """Causal MSA with bounded query tiles and ordinary ND paged caches.

    Each index head selects blocks for its corresponding KV head. Host query
    boundaries come from scheduler metadata; no
    device lengths, expert ids, or scores are read back in this path.
    """
    block_size = key_cache.shape[1]
    if (
        block_size != index_cache.shape[1]
        or query.shape[1] % key_cache.shape[2]
        or index_query.shape[1] != key_cache.shape[2]
    ):
        raise ValueError("Incompatible M3 page sizes or GQA heads")
    actual = query_starts[-1]
    if actual > query.shape[0] or topk_blocks <= 0:
        raise ValueError("Invalid M3 sparse attention metadata")
    # Scheduler padding has no cache entries. Valid request tokens must have
    # real slots; slicing by CPU boundaries avoids device nonzero()/sync.
    for cache, update, slots in (
        (key_cache, key, slot_mapping),
        (value_cache, value, slot_mapping),
        (index_cache, index_key, index_slot_mapping),
    ):
        cache.flatten(0, 1).index_copy_(0, slots[:actual].long(), update[:actual].to(cache.dtype))
    output = torch.zeros_like(query)
    if actual == 0:
        return output
    num_blocks = (max_seq_len + block_size - 1) // block_size
    if min(block_table.shape[1], index_block_table.shape[1]) < num_blocks:
        raise ValueError("Block table does not cover max_seq_len")
    positions = torch.arange(num_blocks * block_size, device=query.device)
    block_ids = torch.arange(num_blocks, device=query.device)
    offsets = torch.arange(block_size, device=query.device)
    repeat = query.shape[1] // key_cache.shape[2]
    for request, (start, stop) in enumerate(zip(query_starts[:-1], query_starts[1:])):
        if start == stop:
            continue
        seq_len = seq_lens[request]
        valid_blocks = block_ids * block_size < seq_len
        pages = torch.where(valid_blocks, block_table[request, :num_blocks], 0).long()
        index_pages = torch.where(valid_blocks, index_block_table[request, :num_blocks], 0).long()
        ik = index_cache.index_select(0, index_pages).flatten(0, 1).float()
        # Uninitialized padding may contain NaNs: mask it before dot products.
        ik = torch.where((positions < seq_len)[:, None], ik, 0)
        for begin in range(start, stop, ATTENTION_QUERY_TILE):
            end = min(stop, begin + ATTENTION_QUERY_TILE)
            qpos = seq_len - (stop - start) + torch.arange(begin - start, end - start, device=query.device)
            scores = torch.matmul(index_query[begin:end].float(), ik.T)
            visible = positions[None, :] <= qpos[:, None]
            scores = scores.masked_fill(~visible[:, None, :], -torch.inf)
            scores = scores.view(end - begin, index_query.shape[1], num_blocks, block_size).amax(-1)
            causal_blocks = block_ids[None, :] <= qpos[:, None] // block_size
            forced_local = block_ids[None, :] > qpos[:, None] // block_size - local_blocks
            selected_heads = []
            # Preserve each head's original top-k call and selected block order.
            # Only the subsequent independent attention batches are combined.
            for head in range(index_query.shape[1]):
                head_scores = torch.where(forced_local & causal_blocks, 1e29, scores[:, head])
                head_scores = torch.where((block_ids[None, :] < init_blocks) & causal_blocks, 1e30, head_scores)
                head_scores = head_scores.masked_fill(~causal_blocks, -torch.inf)
                selected_heads.append(head_scores.topk(min(topk_blocks, num_blocks), -1).indices)
            selected = torch.stack(selected_heads, dim=1)
            logical = (selected[..., None] * block_size + offsets).flatten(2)
            selected_pages = pages[selected]
            heads = torch.arange(index_query.shape[1], device=query.device)[None, :, None]
            k = key_cache[selected_pages, :, heads].flatten(2, 3).float()
            v = value_cache[selected_pages, :, heads].flatten(2, 3).float()
            mask = (logical <= qpos[:, None, None]) & (logical < seq_len)
            k = torch.where(mask[..., None], k, 0)
            v = torch.where(mask[..., None], v, 0)
            batches = (end - begin) * index_query.shape[1]
            q = query[begin:end].float().reshape(batches, repeat, query.shape[-1])
            k = k.reshape(batches, -1, query.shape[-1])
            v = v.reshape_as(k)
            logits = torch.bmm(q, k.transpose(1, 2)) * scale
            probs = logits.masked_fill(~mask.reshape(batches, 1, -1), -torch.inf).softmax(-1)
            output[begin:end] = torch.bmm(probs, v).reshape_as(query[begin:end]).to(query.dtype)
    return output
