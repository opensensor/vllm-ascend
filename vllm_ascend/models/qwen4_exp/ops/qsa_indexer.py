# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-free QSA indexer selection ops (torch, Triton-free).

Ports the *formulas* from the vLLM CUDA fork's
``models/qwen4_exp/nvidia/ops/qsa_indexer.py`` (paged scoring + top-k + expand)
and the mean-pool compression from ``ops/qsa.py`` to plain PyTorch:

1. Compress raw keys into blocks of ``compress_ratio`` by mean pooling; block
   ``m`` covers raw tokens ``[m*ratio, (m+1)*ratio)``.
2. Score each visible compressed block against a query's ``index_n_heads``
   heads: ``score[m] = sum_h relu(q_h . k_m)`` (per-head ReLU, summed) with the
   dot products accumulated in ``accum_dtype``.
3. Deterministically select the top ``block_topk = token_topk // ratio`` visible
   blocks; ties break by ascending block index (a stable descending sort). If
   fewer than ``block_topk`` blocks are visible, all are kept.
4. Expand selected blocks back to token ids and append the causal tail of the
   current (partial) group.

The selection is bitwise-deterministic and matches the T0.6 eager reference
(``tests/ut/qwen38_1m/reference/qsa_indexer_reference.py``).
"""

from __future__ import annotations

import torch

_PAD_INDEX = -1


def compress_keys(
    raw_keys: torch.Tensor,
    compress_ratio: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mean-pool complete groups of ``compress_ratio`` raw keys.

    Args:
        raw_keys: ``[S, D]`` raw key rows (single kv head).
        compress_ratio: tokens per compressed block.
        out_dtype: dtype for the pooled keys (defaults to ``raw_keys.dtype``).

    Returns:
        ``[S // compress_ratio, D]`` compressed keys.
    """
    if raw_keys.ndim != 2:
        raise ValueError("raw_keys must be [S, D]")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    seq_len, dim = raw_keys.shape
    num_blocks = seq_len // compress_ratio
    target_dtype = out_dtype if out_dtype is not None else raw_keys.dtype
    if num_blocks == 0:
        return raw_keys.new_zeros((0, dim), dtype=target_dtype)
    trimmed = raw_keys[: num_blocks * compress_ratio]
    pooled = trimmed.view(num_blocks, compress_ratio, dim).mean(dim=1)
    return pooled.to(target_dtype)


def indexer_block_scores(
    query: torch.Tensor,
    compressed_keys: torch.Tensor,
    *,
    accum_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Relu-summed indexer logits ``sum_h relu(q_{t,h} . k_n)``.

    Args:
        query: ``[T, H, D]`` indexer queries.
        compressed_keys: ``[N, D]`` compressed keys.
        accum_dtype: dtype the dot products / reductions accumulate in (fp32 on
            the 310P path, matching the fork's ``out_dtype=tl.float32``).

    Returns:
        ``[T, N]`` scores.
    """
    if query.ndim != 3:
        raise ValueError("query must be [T, H, D]")
    if compressed_keys.ndim != 2:
        raise ValueError("compressed_keys must be [N, D]")
    q = query.to(accum_dtype)
    k = compressed_keys.to(accum_dtype)
    per_head = torch.einsum("thd,nd->thn", q, k)
    return torch.clamp(per_head, min=0.0).sum(dim=1)


def select_topk_blocks(
    scores_row: torch.Tensor,
    visible_blocks: int,
    block_topk: int,
) -> torch.Tensor:
    """Deterministic top-k block selection over the visible causal range.

    Returns block indices (``int64``) in descending-score order, ties broken by
    ascending block index via a stable descending sort. When fewer than
    ``block_topk`` blocks are visible, every visible block is returned.
    """
    keep = min(int(visible_blocks), int(block_topk))
    if keep <= 0:
        return scores_row.new_empty((0,), dtype=torch.long)
    visible_scores = scores_row[:visible_blocks]
    # Stable descending sort: equal scores keep ascending original (block)
    # index -- identical to sorted(range(n), key=lambda m: (-score[m], m)).
    order = torch.argsort(visible_scores, descending=True, stable=True)
    return order[:keep].to(torch.long)


def expand_block_selection(
    selected_blocks: torch.Tensor,
    pos: int,
    compress_ratio: int,
    token_topk: int,
) -> tuple[torch.Tensor, int]:
    """Expand selected blocks + causal tail into a packed token-index buffer.

    Mirrors ``_expand_qsa_indices_kernel`` / the reference ``expand_selection``.
    Returns ``(packed, valid_count)`` where ``packed`` has length
    ``token_topk + compress_ratio - 1`` and is ``-1``-padded.
    """
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    num_selected = int(selected_blocks.numel())
    complete_blocks = min(num_selected, block_topk)
    expanded_count = complete_blocks * compress_ratio
    tail_start = ((pos + 1) // compress_ratio) * compress_ratio
    tail_count = (pos + 1) - tail_start

    packed = selected_blocks.new_full((output_width,), _PAD_INDEX, dtype=torch.int64)
    if expanded_count > 0:
        blocks = selected_blocks[:complete_blocks].to(torch.int64)
        offsets = torch.arange(compress_ratio, device=selected_blocks.device)
        expanded = (blocks.unsqueeze(1) * compress_ratio + offsets.unsqueeze(0)).reshape(-1)
        packed[:expanded_count] = expanded
    for tail_offset in range(tail_count):
        if tail_offset >= compress_ratio - 1:
            break
        col = expanded_count + tail_offset
        if col < output_width:
            packed[col] = tail_start + tail_offset
    valid_count = expanded_count + tail_count
    return packed, valid_count


def qsa_indexer_select(
    query: torch.Tensor,
    compressed_keys: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
    accum_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score, select, and expand for every query token.

    Args:
        query: ``[T, H, D]`` indexer queries (already cast to the storage dtype).
        compressed_keys: ``[N, D]`` compressed keys (already cast).
        positions: ``[T]`` logical positions of each query token.
        compress_ratio, token_topk: budget geometry.
        accum_dtype: score accumulation dtype.

    Returns:
        ``(packed, valid_counts)`` -- ``packed`` = ``[T, token_topk+ratio-1]``
        (-1 padded token ids) and ``valid_counts`` = ``[T]`` (int64).
    """
    if token_topk % compress_ratio != 0:
        raise ValueError("token_topk must be divisible by compress_ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    seq_len = query.shape[0]
    num_blocks = compressed_keys.shape[0]

    scores = indexer_block_scores(query, compressed_keys, accum_dtype=accum_dtype)
    packed = query.new_full((seq_len, output_width), _PAD_INDEX, dtype=torch.int64)
    valid_counts = query.new_zeros((seq_len,), dtype=torch.int64)
    for t in range(seq_len):
        pos = int(positions[t].item())
        visible = min((pos + 1) // compress_ratio, num_blocks)
        selected = select_topk_blocks(scores[t], visible, block_topk)
        row, count = expand_block_selection(selected, pos, compress_ratio, token_topk)
        packed[t] = row
        valid_counts[t] = count
    return packed, valid_counts


__all__ = [
    "compress_keys",
    "expand_block_selection",
    "indexer_block_scores",
    "qsa_indexer_select",
    "select_topk_blocks",
]
