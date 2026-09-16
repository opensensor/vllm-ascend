# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for the DeepSeek V4.1 sparse-attention indexer.

Ports the *formulas* (never the Triton kernels) from the fork
``vllm/models/deepseek_v41/attention.py`` (the Lightning-Indexer query/key
projection, ``weights_proj`` per-head weights, ``softmax_scale`` and top-k
selection) and ``compressor.py`` (the CSA2 mean-pool compression of
``compress_ratio`` consecutive tokens into one latent).

Pipeline for one causal sequence:

1. **Compress** raw indexer keys into blocks of ``compress_ratio`` by mean
   pooling; block ``m`` pools raw tokens ``[m*ratio, (m+1)*ratio)`` (CSA2 ratio
   1 or 2 in production).
2. **Score** each visible compressed block against the query's ``index_n_heads``
   heads with the DeepSeek Lightning-Indexer form:
   ``score[t, m] = sum_h w[t,h] * relu(softmax_scale * (q[t,h] . k_block_m))``,
   where ``w`` are the per-token per-head ``weights_proj`` outputs.
3. Visible blocks are the fully-formed causal groups: ``(pos+1) // ratio``.
4. **Select** the top ``block_topk = index_topk // ratio`` visible blocks; if
   fewer are visible, all are kept (short-context dense fallback).

The self-consistency oracles are a brute-force einsum re-scoring and a
brute-force argsort selection; both must match the vectorized path exactly (set
equality on the selected block ids).
"""

from __future__ import annotations

import torch

# DeepSeek V4.1 indexer geometry (config: index_n_heads, index_head_dim,
# index_topk). Small faithful defaults for the reference.
INDEXER_N_HEADS = 4
INDEXER_HEAD_DIM = 128
INDEXER_TOPK = 8  # token budget; block_topk = INDEXER_TOPK // compress_ratio


def compress_keys(raw_keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Mean-pool complete groups of ``compress_ratio`` raw keys (CSA2).

    Args:
        raw_keys: ``[S, D]`` raw indexer key rows (single index kv head).
        compress_ratio: tokens pooled per compressed block (1 or 2).

    Returns:
        ``[S // compress_ratio, D]`` float64 compressed keys.
    """
    raw_keys = raw_keys.double()
    seq_len, dim = raw_keys.shape
    num_blocks = seq_len // compress_ratio
    if num_blocks == 0:
        return raw_keys.new_zeros((0, dim))
    trimmed = raw_keys[: num_blocks * compress_ratio]
    return trimmed.view(num_blocks, compress_ratio, dim).mean(dim=1)


def bruteforce_compress_keys(raw_keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Explicit per-block averaging (test oracle)."""
    raw_keys = raw_keys.double()
    seq_len, dim = raw_keys.shape
    num_blocks = seq_len // compress_ratio
    out = torch.empty(num_blocks, dim, dtype=torch.float64)
    for m in range(num_blocks):
        acc = torch.zeros(dim, dtype=torch.float64)
        for j in range(compress_ratio):
            acc = acc + raw_keys[m * compress_ratio + j]
        out[m] = acc / compress_ratio
    return out


def visible_block_count(pos: int, compress_ratio: int) -> int:
    """Number of fully-formed causal compressed blocks for position ``pos``."""
    return (pos + 1) // compress_ratio


def indexer_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    compressed_keys: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Lightning-Indexer logits: weighted relu-summed dot products.

    Args:
        query: ``[T, H, D]`` indexer queries.
        weights: ``[T, H]`` per-head ``weights_proj`` weights.
        compressed_keys: ``[N, D]`` compressed keys.
        softmax_scale: dot-product scale (``index_head_dim ** -0.5``).

    Returns:
        ``[T, N]`` scores ``sum_h w[t,h] * relu(scale * (q_{t,h} . k_n))``.
    """
    query = query.double()
    weights = weights.double()
    compressed_keys = compressed_keys.double()
    per_head = torch.einsum("thd,nd->thn", query, compressed_keys) * softmax_scale
    per_head = torch.clamp(per_head, min=0.0)
    return torch.einsum("thn,th->tn", per_head, weights)


def bruteforce_indexer_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    compressed_keys: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Per-(token, block, head) loop scoring (test oracle)."""
    query = query.double()
    weights = weights.double()
    compressed_keys = compressed_keys.double()
    seq_len, num_heads, _ = query.shape
    num_blocks = compressed_keys.shape[0]
    out = torch.zeros(seq_len, num_blocks, dtype=torch.float64)
    for t in range(seq_len):
        for m in range(num_blocks):
            acc = 0.0
            for h in range(num_heads):
                dot = float((query[t, h] * compressed_keys[m]).sum()) * softmax_scale
                acc += float(weights[t, h]) * max(dot, 0.0)
            out[t, m] = acc
    return out


def select_blocks(
    scores_row: torch.Tensor,
    visible_blocks: int,
    block_topk: int,
) -> list[int]:
    """Deterministic top-k block selection over the visible causal range.

    Returns block indices in descending-score order, ties broken by ascending
    block index. If fewer than ``block_topk`` blocks are visible, all are kept.
    """
    keep = min(visible_blocks, block_topk)
    if keep <= 0:
        return []
    visible_scores = scores_row[:visible_blocks]
    order = sorted(range(visible_blocks), key=lambda m: (-visible_scores[m].item(), m))
    return order[:keep]


def bruteforce_select_blocks(
    scores_row: torch.Tensor,
    visible_blocks: int,
    block_topk: int,
) -> set[int]:
    """Independent argmax-peeling selection returning a *set* (test oracle)."""
    keep = min(visible_blocks, block_topk)
    remaining = list(range(visible_blocks))
    chosen: set[int] = set()
    for _ in range(keep):
        best = max(remaining, key=lambda m: (scores_row[m].item(), -m))
        chosen.add(best)
        remaining.remove(best)
    return chosen


def indexer_select(
    query: torch.Tensor,
    raw_keys: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    *,
    compress_ratio: int,
    index_topk: int = INDEXER_TOPK,
    softmax_scale: float | None = None,
) -> list[list[int]]:
    """Full indexer path for one sequence: compress, score, select per token.

    Args:
        query: ``[T, H, D]`` indexer queries.
        raw_keys: ``[S, D]`` raw keys for the whole causal context.
        positions: ``[T]`` logical positions of each query token.
        weights: ``[T, H]`` per-head weights.
        compress_ratio, index_topk: CSA2 ratio and token budget.
        softmax_scale: defaults to ``D ** -0.5``.

    Returns:
        ``[T]`` lists of selected block ids (descending score order).
    """
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    compressed = compress_keys(raw_keys, compress_ratio)
    scores = indexer_scores(query, weights, compressed, softmax_scale)
    block_topk = index_topk // compress_ratio
    selected = []
    for t in range(query.shape[0]):
        pos = int(positions[t].item())
        vis = min(visible_block_count(pos, compress_ratio), scores.shape[1])
        selected.append(select_blocks(scores[t], vis, block_topk))
    return selected


__all__ = [
    "INDEXER_N_HEADS",
    "INDEXER_HEAD_DIM",
    "INDEXER_TOPK",
    "compress_keys",
    "bruteforce_compress_keys",
    "visible_block_count",
    "indexer_scores",
    "bruteforce_indexer_scores",
    "select_blocks",
    "bruteforce_select_blocks",
    "indexer_select",
]
