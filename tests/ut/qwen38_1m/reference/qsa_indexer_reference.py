# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for the Qwen4Exp QSA indexer.

Ports the *formulas* (not Triton) from
``vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`` (scoring + top-k selection +
expand) and ``ops/qsa.py`` (``_compress_qsa_groups_kernel``, mean pooling).

Pipeline (per query token at logical position ``pos``):
  1. Compress raw keys into blocks of ``compress_ratio`` by mean pooling; block
     ``m`` covers raw tokens ``[m*ratio, (m+1)*ratio)``.
  2. Score each visible compressed block against the query's ``index_n_heads``
     heads: ``score[m] = sum_h relu(q_h . k_block_m)`` (per-head ReLU, summed).
  3. Visible blocks are the fully-formed causal groups: ``(pos+1) // ratio``.
  4. Select the top ``block_topk = token_topk // ratio`` visible blocks; if fewer
     than ``block_topk`` are visible, all are kept (no dropping).
  5. Expand selected blocks back to token ids and append the causal tail of the
     current (partial) group. ``valid_count = complete_blocks*ratio + tail_len``.

The 2,048-token budget is ``token_topk``: with ``ratio=4`` -> ``block_topk=512``.
For contexts <= budget, selection is the full causal prefix (dense); the budget
only prunes once ``visible_blocks > block_topk``.
"""

from __future__ import annotations

import torch

# Default indexer geometry from the plan (T6.1): 4 q-heads, 1 k-head, dim 128,
# compression ratio 4, budget 2048.
INDEXER_N_HEADS = 4
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 128
INDEXER_COMPRESS_RATIO = 4
INDEXER_BUDGET = 2048

_PAD_INDEX = -1


def compress_keys(raw_keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Mean-pool complete groups of ``compress_ratio`` raw keys.

    Args:
        raw_keys: ``[S, D]`` raw key rows (single kv head).
        compress_ratio: tokens per compressed block.

    Returns:
        ``[num_blocks, D]`` compressed keys; ``num_blocks = S // compress_ratio``.
    """
    raw_keys = raw_keys.double()
    seq_len, dim = raw_keys.shape
    num_blocks = seq_len // compress_ratio
    if num_blocks == 0:
        return raw_keys.new_zeros((0, dim))
    trimmed = raw_keys[: num_blocks * compress_ratio]
    return trimmed.view(num_blocks, compress_ratio, dim).mean(dim=1)


def visible_block_count(pos: int, compress_ratio: int) -> int:
    """Number of fully-formed causal compressed blocks for position ``pos``."""
    return (pos + 1) // compress_ratio


def indexer_scores(
    query: torch.Tensor,
    compressed_keys: torch.Tensor,
) -> torch.Tensor:
    """Relu-summed indexer logits.

    Args:
        query: ``[T, H, D]`` indexer queries.
        compressed_keys: ``[N, D]`` compressed keys.

    Returns:
        ``[T, N]`` scores ``sum_h relu(q_{t,h} . k_n)``.
    """
    query = query.double()
    compressed_keys = compressed_keys.double()
    # [T, H, N] per-head dot, ReLU, then sum over heads.
    per_head = torch.einsum("thd,nd->thn", query, compressed_keys)
    return torch.clamp(per_head, min=0.0).sum(dim=1)


def select_blocks(
    scores_row: torch.Tensor,
    visible_blocks: int,
    block_topk: int,
) -> list[int]:
    """Deterministic top-k block selection over the visible range.

    Returns block indices in descending-score order (ties broken by ascending
    block index, matching a stable selection). If fewer than ``block_topk``
    blocks are visible, every visible block is returned.
    """
    keep = min(visible_blocks, block_topk)
    if keep <= 0:
        return []
    visible_scores = scores_row[:visible_blocks]
    # Stable deterministic order: sort by (-score, index).
    order = sorted(range(visible_blocks), key=lambda m: (-visible_scores[m].item(), m))
    return order[:keep]


def expand_selection(
    selected_blocks: list[int],
    pos: int,
    compress_ratio: int,
    token_topk: int,
) -> tuple[torch.Tensor, int]:
    """Expand selected blocks + causal tail into a packed token-index buffer.

    Mirrors ``_expand_qsa_indices_kernel``. Returns ``(packed, valid_count)``
    where ``packed`` has length ``token_topk + compress_ratio - 1`` and is
    ``-1``-padded, and ``valid_count`` is the tile-loop bound.
    """
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    complete_blocks = min(len(selected_blocks), block_topk)
    expanded_count = complete_blocks * compress_ratio
    tail_start = ((pos + 1) // compress_ratio) * compress_ratio
    tail_count = (pos + 1) - tail_start

    packed = torch.full((output_width,), _PAD_INDEX, dtype=torch.int64)
    for col in range(output_width):
        if col < expanded_count:
            block_rank = col // compress_ratio
            offset = col % compress_ratio
            packed[col] = selected_blocks[block_rank] * compress_ratio + offset
        else:
            tail_offset = col - expanded_count
            if tail_offset < tail_count and tail_offset < compress_ratio - 1:
                packed[col] = tail_start + tail_offset
            # else stays -1
    valid_count = expanded_count + tail_count
    return packed, valid_count


def qsa_select_tokens(
    query: torch.Tensor,
    raw_keys: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
    token_topk: int = INDEXER_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full indexer path for one sequence.

    Args:
        query: ``[T, H, D]`` indexer queries.
        raw_keys: ``[S, D]`` raw keys for the whole (causal) context.
        positions: ``[T]`` logical positions of each query token.
        compress_ratio, token_topk: budget geometry.

    Returns:
        ``(packed, valid_counts)`` with ``packed`` = ``[T, token_topk+ratio-1]``
        (-1 padded token ids) and ``valid_counts`` = ``[T]``.
    """
    compressed = compress_keys(raw_keys, compress_ratio)
    scores = indexer_scores(query, compressed)  # [T, N]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    seq_len = query.shape[0]
    packed = torch.full((seq_len, output_width), _PAD_INDEX, dtype=torch.int64)
    valid_counts = torch.zeros(seq_len, dtype=torch.int64)
    for t in range(seq_len):
        pos = int(positions[t].item())
        vis = visible_block_count(pos, compress_ratio)
        vis = min(vis, scores.shape[1])
        selected = select_blocks(scores[t], vis, block_topk)
        row, count = expand_selection(selected, pos, compress_ratio, token_topk)
        packed[t] = row
        valid_counts[t] = count
    return packed, valid_counts


def selected_token_set(packed_row: torch.Tensor) -> set[int]:
    """The set of valid (non -1) token ids in a packed selection row."""
    return {int(v) for v in packed_row.tolist() if v >= 0}


__all__ = [
    "INDEXER_N_HEADS",
    "INDEXER_KV_HEADS",
    "INDEXER_HEAD_DIM",
    "INDEXER_COMPRESS_RATIO",
    "INDEXER_BUDGET",
    "compress_keys",
    "visible_block_count",
    "indexer_scores",
    "select_blocks",
    "expand_selection",
    "qsa_select_tokens",
    "selected_token_set",
]
