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

from dataclasses import dataclass

import torch

_PAD_INDEX = -1
_QSA_INDEX_CACHE_SCRATCH_ROWS = 3


@dataclass(frozen=True)
class QSAGroupSelection:
    """Device-resident QSA selection before expansion to token ids.

    Keeping the learned selection in compression-group form avoids expanding
    512 selected groups into roughly 2K token ids.  The native sparse
    attention kernel expands each group while reading the paged KV cache.
    """

    group_indices: torch.Tensor
    group_counts: torch.Tensor
    tail_starts: torch.Tensor
    tail_counts: torch.Tensor


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
    selection = qsa_indexer_select_groups(
        query,
        compressed_keys,
        positions,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        accum_dtype=accum_dtype,
    )
    return expand_group_selection(selection, compress_ratio, token_topk)


def qsa_indexer_select_groups(
    query: torch.Tensor,
    compressed_keys: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
    accum_dtype: torch.dtype = torch.float32,
) -> QSAGroupSelection:
    """Select compressed QSA groups without host synchronization.

    All visibility masking, sorting, and tail metadata stay on device.  This is
    the contract consumed by the planned 310P sparse-attention kernel; unlike
    :func:`qsa_indexer_select`, it never materializes per-token indices.
    """
    if token_topk % compress_ratio != 0:
        raise ValueError("token_topk must be divisible by compress_ratio")
    if positions.ndim != 1 or positions.shape[0] != query.shape[0]:
        raise ValueError("positions must be [T] aligned with the queries")

    block_topk = token_topk // compress_ratio
    seq_len = query.shape[0]
    num_blocks = compressed_keys.shape[0]
    selected_width = min(block_topk, num_blocks)
    positions_long = positions.to(torch.long)
    visible_blocks = torch.div(
        positions_long + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp_max(num_blocks)

    if selected_width == 0:
        selected = query.new_empty((seq_len, 0), dtype=torch.long)
    else:
        scores = indexer_block_scores(query, compressed_keys, accum_dtype=accum_dtype)
        block_ids = torch.arange(num_blocks, device=query.device)
        visible_mask = block_ids.unsqueeze(0) < visible_blocks.unsqueeze(1)
        masked_scores = scores.masked_fill(~visible_mask, -torch.inf)
        # Stable descending order preserves the reference tie-break: lower
        # compression-group id wins when scores are equal.
        selected = torch.argsort(masked_scores, dim=1, descending=True, stable=True)[:, :selected_width]

    group_counts = visible_blocks.clamp_max(selected_width)
    tail_starts = (
        torch.div(
            positions_long + 1,
            compress_ratio,
            rounding_mode="floor",
        )
        * compress_ratio
    )
    tail_counts = positions_long + 1 - tail_starts
    return QSAGroupSelection(
        group_indices=selected,
        group_counts=group_counts,
        tail_starts=tail_starts,
        tail_counts=tail_counts,
    )


def qsa_indexer_select_groups_310(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
) -> QSAGroupSelection:
    """Select learned QSA groups through the dedicated 310P score kernel.

    ``compressed_key_cache`` is paged ND storage
    ``[physical_blocks, groups_per_block + 3, index_head_dim]``. The final
    three rows are private scratch storage used to carry an incomplete
    four-token group across scheduler invocations. The native kernel
    performs page translation and the four-head ReLU-summed dot products in a
    single launch. Stable sorting remains a device operation so equal scores
    retain the reference's lower-group-id tie break.
    """
    if query.device.type != "npu":
        raise RuntimeError("qsa_indexer_select_groups_310 is an Ascend NPU-only path")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    if query.ndim != 3 or compressed_key_cache.ndim != 3:
        raise ValueError("query/cache must be [T,H,D] and [blocks,rows,D]")
    if query.shape[-1] != compressed_key_cache.shape[-1]:
        raise ValueError("query and compressed cache head dimensions differ")
    if positions.shape != (query.shape[0],):
        raise ValueError("positions must be [T]")
    op_namespace = getattr(torch.ops, "_C_ascend", None)
    op = None if op_namespace is None else getattr(op_namespace, "npu_qsa_indexer_score_310", None)
    if op is None:
        raise RuntimeError("vLLM Ascend was built without the dedicated 310P QSA index-score operator")

    scores = op(
        query.contiguous(),
        compressed_key_cache.contiguous(),
        block_table.to(dtype=torch.int32).contiguous(),
        query_start_loc.to(dtype=torch.int32).contiguous(),
        positions.to(dtype=torch.int32).contiguous(),
        compress_ratio,
    )
    block_topk = token_topk // compress_ratio
    selected_width = min(block_topk, scores.shape[1])
    selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :selected_width]
    positions_long = positions.to(torch.long)
    visible_groups = torch.div(positions_long + 1, compress_ratio, rounding_mode="floor").clamp_max(scores.shape[1])
    group_counts = visible_groups.clamp_max(selected_width)
    tail_starts = torch.div(positions_long + 1, compress_ratio, rounding_mode="floor") * compress_ratio
    tail_counts = positions_long + 1 - tail_starts
    return QSAGroupSelection(selected, group_counts, tail_starts, tail_counts)


def qsa_indexer_score_310_reference(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int = 4,
) -> torch.Tensor:
    """Host reference for the 310P paged index-score operator.

    The final three rows of every physical page are cache-update scratch and
    are intentionally excluded from the logical group address space.
    """
    if query.device.type != "cpu" or compressed_key_cache.device.type != "cpu":
        raise ValueError("the 310P score reference expects CPU tensors")
    if query.ndim != 3 or compressed_key_cache.ndim != 3:
        raise ValueError("query/cache must be [T,H,D] and [blocks,rows,D]")
    groups_per_block = compressed_key_cache.shape[1] - _QSA_INDEX_CACHE_SCRATCH_ROWS
    if groups_per_block <= 0:
        raise ValueError("compressed cache has no logical group rows")
    max_groups = block_table.shape[1] * groups_per_block
    scores = torch.full((query.shape[0], max_groups), -torch.inf, dtype=torch.float32)
    boundaries = query_start_loc.to(dtype=torch.int64).tolist()
    for row in range(query.shape[0]):
        request = next(request for request in range(len(boundaries) - 1) if row < boundaries[request + 1])
        visible_groups = min((int(positions[row]) + 1) // compress_ratio, max_groups)
        for group in range(visible_groups):
            logical_block, group_row = divmod(group, groups_per_block)
            physical_block = int(block_table[request, logical_block])
            key = compressed_key_cache[physical_block, group_row].float()
            dots = torch.matmul(query[row].float(), key)
            scores[row, group] = torch.clamp(dots, min=0).sum()
    return scores


def expand_group_selection(
    selection: QSAGroupSelection,
    compress_ratio: int,
    token_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized compatibility expansion for the torch reference path."""
    output_width = token_topk + compress_ratio - 1
    seq_len, selected_width = selection.group_indices.shape
    device = selection.group_indices.device
    packed = selection.group_indices.new_full((seq_len, output_width), _PAD_INDEX)

    if selected_width > 0:
        offsets = torch.arange(compress_ratio, device=device)
        expanded = (selection.group_indices.unsqueeze(-1) * compress_ratio + offsets.view(1, 1, -1)).reshape(
            seq_len, selected_width * compress_ratio
        )
        expanded_ranks = torch.arange(selected_width * compress_ratio, device=device)
        expanded_valid = expanded_ranks.unsqueeze(0) < (selection.group_counts * compress_ratio).unsqueeze(1)
        packed[:, : selected_width * compress_ratio] = torch.where(
            expanded_valid,
            expanded,
            expanded.new_full((), _PAD_INDEX),
        )

    tail_offsets = torch.arange(compress_ratio - 1, device=device)
    tail_tokens = selection.tail_starts.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = tail_offsets.unsqueeze(0) < selection.tail_counts.unsqueeze(1)
    tail_columns = selection.group_counts.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
    rows = torch.arange(seq_len, device=device).unsqueeze(1).expand_as(tail_columns)
    valid_rows = rows[tail_valid]
    valid_columns = tail_columns[tail_valid]
    valid_tokens = tail_tokens[tail_valid]
    packed[valid_rows, valid_columns] = valid_tokens
    valid_counts = selection.group_counts * compress_ratio + selection.tail_counts
    return packed, valid_counts


__all__ = [
    "compress_keys",
    "expand_group_selection",
    "expand_block_selection",
    "indexer_block_scores",
    "qsa_indexer_select",
    "qsa_indexer_select_groups",
    "qsa_indexer_select_groups_310",
    "QSAGroupSelection",
    "select_topk_blocks",
]
