# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device tensor primitives for GLM's pooled sparse-attention indexer."""

import torch

# A 32-bit score temporary stays at or below 256 MiB before top-k selection.
MAX_KPOOL_SCORE_ELEMENTS = 1 << 26


def hadamard128(rows: torch.Tensor) -> torch.Tensor:
    """Apply the normalized 128-wide Walsh-Hadamard rotation in FP32."""
    if rows.shape[-1] != 128:
        raise ValueError(f"GLM kpool requires 128-wide index keys, got {rows.shape[-1]}")
    rotated = rows.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = rotated.reshape(*rotated.shape[:-1], 128 // (2 * stride), 2, stride)
        rotated = torch.stack(
            (pairs[..., 0, :] + pairs[..., 1, :], pairs[..., 0, :] - pairs[..., 1, :]),
            dim=-2,
        ).reshape_as(rotated)
    return rotated * (128**-0.5)


def compress_kpool(
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Compress full pools using GLM's per-dimension gated softmax."""
    if keys.shape != gate_scores.shape or keys.shape[-2:] != ape.shape:
        raise ValueError("K, gate, and APE shapes disagree for GLM kpool compression")
    probabilities = torch.softmax(gate_scores.float() + ape.float(), dim=-2)
    pooled = (keys.float() * probabilities).sum(dim=-2).to(torch.bfloat16)
    return hadamard128(pooled).to(torch.bfloat16)


def score_kpool(
    queries: torch.Tensor,
    weights: torch.Tensor,
    pooled_keys: torch.Tensor,
) -> torch.Tensor:
    """Head-weighted ReLU MQA logits over rotated, compressed keys."""
    if queries.ndim != 3 or queries.shape[-1] != 128:
        raise ValueError("GLM kpool queries must be [tokens, heads, 128]")
    if weights.shape != queries.shape[:2] or pooled_keys.shape[-1] != 128:
        raise ValueError("GLM kpool score inputs have inconsistent shapes")
    num_queries, num_heads, _ = queries.shape
    q_rot = hadamard128(queries).to(torch.bfloat16)
    logits = q_rot.reshape(-1, 128).float() @ pooled_keys.float().transpose(0, 1)
    logits = logits.reshape(num_queries, num_heads, -1).relu_()
    # Logits are private scratch. Reuse them instead of allocating a second
    # head-by-query-by-pool tensor (up to another 256 MiB in prefill).
    return logits.mul_(weights.float().unsqueeze(-1)).sum(dim=1)


def topk_pool_indices(
    logits: torch.Tensor,
    count: int,
    *,
    rows_per_call: int | None = None,
) -> torch.Tensor:
    """Select pools, optionally bounding the row batch for diagnostic trials.

    Columns are never partitioned: each row retains its complete candidate
    set. The default preserves the qualified top-k dispatch geometry.
    """
    if rows_per_call is None:
        return torch.topk(logits, count, dim=1).indices.to(torch.int32)
    if rows_per_call <= 0:
        raise ValueError("top-k rows_per_call must be positive")
    result = torch.empty((logits.shape[0], count), dtype=torch.int32, device=logits.device)
    for start in range(0, logits.shape[0], rows_per_call):
        end = min(start + rows_per_call, logits.shape[0])
        result[start:end] = torch.topk(logits[start:end], count, dim=1).indices.to(torch.int32)
    return result


def select_kpool_groups(
    logits: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
    *,
    scores_are_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select completed pools and retain the current incomplete pool as tail.

    ``scores_are_causal`` is reserved for producers that already wrote -inf
    to every incomplete/unused pool. It avoids another full-width mask/copy;
    index/rank checks on top-k padding remain required on 310P.
    """
    if logits.ndim != 2 or positions.ndim != 1 or logits.shape[0] != positions.shape[0]:
        raise ValueError("GLM kpool logits and positions must have matching rows")
    if pool_size <= 1 or topk_tokens <= 0 or topk_tokens % pool_size:
        raise ValueError("GLM kpool topk_tokens must be divisible by pool_size > 1")
    pool_budget = topk_tokens // pool_size
    sequence_lengths = positions.to(torch.int32) + 1
    completed = torch.div(sequence_lengths, pool_size, rounding_mode="floor")
    selected = torch.full((logits.shape[0], pool_budget), -1, dtype=torch.int32, device=logits.device)
    count = min(pool_budget, logits.shape[1])
    if count:
        if count == logits.shape[1]:
            # Every candidate fits. Avoid a top-k and GatherElements on 310P.
            candidate_ids = torch.arange(logits.shape[1], device=logits.device, dtype=torch.int32)
            valid = candidate_ids[None, :] < completed[:, None]
            candidate_rows = candidate_ids[None, :].expand(logits.shape[0], -1)
            selected[:, :count] = torch.where(valid, candidate_rows, -1)
        else:
            if scores_are_causal:
                masked = logits
            else:
                candidate_ids = torch.arange(logits.shape[1], device=logits.device, dtype=torch.int32)
                valid = candidate_ids[None, :] < completed[:, None]
                masked = logits.masked_fill(~valid, -torch.inf)
            topk = topk_pool_indices(masked, count)
            # Some 310P top-k implementations leave arbitrary indices in
            # entries whose score is -inf. An invalid index may even alias a
            # valid pool. Only the first completed ranks can contain pools.
            ranks = torch.arange(count, device=logits.device, dtype=torch.int32)
            valid_selection = (ranks[None, :] < completed[:, None]) & (topk >= 0) & (topk < completed[:, None])
            selected[:, :count] = torch.where(valid_selection, topk, -1)
    group_counts = completed.clamp(max=pool_budget)
    tail_starts = completed * pool_size
    tail_counts = sequence_lengths - tail_starts
    return selected, group_counts, tail_starts, tail_counts


def expand_kpool_groups(
    selected: torch.Tensor,
    tail_starts: torch.Tensor,
    tail_counts: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Expand pool IDs to token IDs and append 0..pool_size-1 tail tokens."""
    offsets = torch.arange(pool_size, device=selected.device, dtype=torch.int32)
    expanded = selected[:, :, None] * pool_size + offsets[None, None, :]
    expanded = expanded.masked_fill(selected[:, :, None] < 0, -1).flatten(1)
    tail_offsets = offsets[:-1]
    tail = tail_starts[:, None] + tail_offsets[None, :]
    tail = tail.masked_fill(tail_offsets[None, :] >= tail_counts[:, None], -1)
    return torch.cat((expanded, tail), dim=1)


def dense_kpool_token_indices(
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> torch.Tensor:
    """Select every causal token when all completed pools fit the budget.

    Keep the same fixed-width layout as ``expand_kpool_groups``: completed
    pools occupy the prefix, and the incomplete pool occupies the final
    ``pool_size - 1`` columns. This avoids score construction and top-k for
    short contexts while preserving the sparse-attention index contract. The
    caller verifies the length limit from host scheduler metadata.
    """
    if positions.ndim != 1 or pool_size <= 1 or topk_tokens <= 0 or topk_tokens % pool_size:
        raise ValueError("GLM kpool dense selection needs 1-D positions and a divisible token budget")
    sequence_lengths = positions.to(torch.int32) + 1
    completed_tokens = torch.div(sequence_lengths, pool_size, rounding_mode="floor") * pool_size
    prefix_offsets = torch.arange(topk_tokens, device=positions.device, dtype=torch.int32)
    prefix = prefix_offsets[None, :].expand(positions.shape[0], -1)
    prefix = prefix.masked_fill(prefix_offsets[None, :] >= completed_tokens[:, None], -1)
    tail_offsets = torch.arange(pool_size - 1, device=positions.device, dtype=torch.int32)
    tail = completed_tokens[:, None] + tail_offsets[None, :]
    tail = tail.masked_fill(tail_offsets[None, :] >= sequence_lengths[:, None] - completed_tokens[:, None], -1)
    return torch.cat((prefix, tail), dim=1)


def score_and_select_kpool_tokens(
    queries: torch.Tensor,
    weights: torch.Tensor,
    pooled_keys: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> torch.Tensor:
    """Bound the head-by-query score temporary for long contexts."""
    if queries.shape[0] == 0:
        return torch.empty((0, topk_tokens + pool_size - 1), dtype=torch.int32, device=queries.device)
    score_elements_per_query = queries.shape[1] * pooled_keys.shape[0]
    query_chunk_size = max(1, MAX_KPOOL_SCORE_ELEMENTS // max(1, score_elements_per_query))
    # Every query subchunk reads the same key bank. Keep one conversion live
    # instead of repeatedly widening all pooled keys as the context grows.
    pooled_keys = pooled_keys.float()
    expanded_chunks = []
    for start in range(0, queries.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, queries.shape[0])
        logits = score_kpool(queries[start:end], weights[start:end], pooled_keys)
        selected, _, tail_starts, tail_counts = select_kpool_groups(
            logits, positions[start:end], topk_tokens, pool_size
        )
        expanded_chunks.append(expand_kpool_groups(selected, tail_starts, tail_counts, pool_size))
    return expanded_chunks[0] if len(expanded_chunks) == 1 else torch.cat(expanded_chunks, dim=0)
