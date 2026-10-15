# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline candidate: bucket QSA score capacity before capturing selection.

This module does not install a model replacement. A caller must admit host
metadata before EVERY graph replay, retire an incompatible capture, and stage
current block tables/query boundaries/causal positions into stable device
buffers. The descriptor includes the scoring backend and packed row layout.
The existing 262144-token limit is independent of an individual bucket.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.models.qwen4_exp.ops import qsa_indexer as _qsa_indexer
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    QSA_SELECTION_FAST_TOPK,
    QSA_SELECTION_POLICIES,
    QSA_SELECTION_STABLE_ARGSORT,
    QSAGroupSelection,
    _use_qsa_matmul_score_batch,
    copy_group_selection_into,
    qsa_indexer_select_groups_310,
)


def _qsa_position_geometry(positions, compress_ratio, capacity):
    """Resolve the canonical helper at capture time, including after hot reload.

    Keep this forwarding name because existing prefix_selection function
    objects retain this module's globals when importlib.reload updates it.
    A directly imported helper would bypass later canonical replacements.
    """
    return _qsa_indexer._qsa_position_geometry(positions, compress_ratio, capacity)


@dataclass(frozen=True)
class SelectionPlan:
    """Immutable capture geometry; no graph handles or replay counters."""

    capacity_blocks: int
    groups_per_block: int
    query_lens: tuple[int, ...]
    compress_ratio: int
    token_topk: int
    max_context: int
    max_matmul_decode_tokens: int
    backend: str
    selection_policy: str

    @property
    def capacity_groups(self):
        return self.capacity_blocks * self.groups_per_block

    def admits(self, sequence_lengths, query_lengths):
        """Host-only pre-dispatch guard, never called from a captured graph."""
        lengths, queries = _host_lengths(sequence_lengths, query_lengths, self.max_context)
        if queries != self.query_lens:
            return False
        actual_blocks = _required_blocks(lengths, self.compress_ratio, self.groups_per_block)
        if actual_blocks > self.capacity_blocks:
            return False
        if self.backend != "prefix" and self.selection_policy == QSA_SELECTION_FAST_TOPK:
            # Unstable top-k can choose different cutoff ties at a new width.
            if actual_blocks != self.capacity_blocks:
                return False
        backend = _backend(
            actual_blocks * self.groups_per_block,
            queries,
            self.token_topk // self.compress_ratio,
            self.max_matmul_decode_tokens,
        )
        return backend == self.backend


def _host_lengths(sequence_lengths, query_lengths, max_context):
    def values(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError("admission requires one-dimensional scheduler CPU integer lengths")
            return tuple(value.tolist())
        result = tuple(value)
        if any(type(length) is not int for length in result):
            raise ValueError("admission lengths must be integers")
        return result

    lengths, queries = values(sequence_lengths), values(query_lengths)
    if not lengths or len(lengths) != len(queries) or sum(queries) <= 0:
        raise ValueError("lengths must describe a nonempty packed batch")
    if any(length < 0 or length > max_context for length in lengths):
        raise ValueError("sequence length exceeds the configured context")
    # Padded rows can have query length > sequence length; this gives negative
    # causal positions and is deliberately preserved by the selection math.
    if any(length < 0 for length in queries):
        raise ValueError("query lengths must be nonnegative")
    return lengths, queries


def _required_blocks(lengths, compress_ratio, groups_per_block):
    complete_groups = max(lengths) // compress_ratio
    return max(1, (complete_groups + groups_per_block - 1) // groups_per_block)


def _backend(capacity, queries, selected_groups, max_decode_tokens):
    if capacity <= selected_groups:
        # The existing _repair_native_group_indices returns the complete
        # ascending prefix for every row in this range, irrespective of score.
        return "prefix"
    use_matmul = _use_qsa_matmul_score_batch(
        queries, sum(queries), capacity, len(queries), len(queries) + 1, max_decode_tokens
    )
    return "matmul" if use_matmul else "native"


def make_plan(
    sequence_lengths,
    query_lengths,
    *,
    table_blocks,
    groups_per_block,
    compress_ratio=4,
    token_topk=2048,
    max_context=262144,
    max_matmul_decode_tokens=2,
    selection_policy=QSA_SELECTION_STABLE_ARGSORT,
):
    """Choose at most 2x visible capacity without crossing a math boundary."""
    if min(table_blocks, groups_per_block, compress_ratio, token_topk, max_context) <= 0:
        raise ValueError("QSA geometry must be positive")
    if token_topk % compress_ratio or max_matmul_decode_tokens < 0:
        raise ValueError("invalid QSA selection or decode geometry")
    if selection_policy not in QSA_SELECTION_POLICIES:
        raise ValueError("unsupported QSA selection policy")
    lengths, queries = _host_lengths(sequence_lengths, query_lengths, max_context)
    required = _required_blocks(lengths, compress_ratio, groups_per_block)
    if required > table_blocks:
        raise ValueError("block table does not cover the visible context")
    selected_groups = token_topk // compress_ratio
    backend = _backend(required * groups_per_block, queries, selected_groups, max_matmul_decode_tokens)
    capacity = min(1 << (required - 1).bit_length(), table_blocks)
    # Round down within this backend's band. Native/GEMM crossover belongs to
    # the existing selector; do not silently change score accumulation paths.
    while (
        capacity > required
        and _backend(capacity * groups_per_block, queries, selected_groups, max_matmul_decode_tokens) != backend
    ):
        capacity -= 1
    if backend != "prefix" and selection_policy == QSA_SELECTION_FAST_TOPK:
        capacity = required
    return SelectionPlan(
        capacity,
        groups_per_block,
        queries,
        compress_ratio,
        token_topk,
        max_context,
        max_matmul_decode_tokens,
        backend,
        selection_policy,
    )


def prefix_selection(positions, *, compress_ratio, token_topk, capacity_groups):
    """Capture-safe complete-prefix selection, only for an admitted bucket."""
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("prefix selection requires positive divisible geometry")
    width = token_topk // compress_ratio
    if not 0 < capacity_groups <= width:
        raise ValueError("prefix selection requires capacity within the exact group budget")
    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional logical causal positions, not MRoPE axes")
    visible, tail_starts, tail_counts = _qsa_position_geometry(positions, compress_ratio, capacity_groups)
    groups = torch.arange(width, dtype=torch.int32, device=positions.device).expand(positions.numel(), -1).contiguous()
    return QSAGroupSelection(groups, visible.to(torch.int32), tail_starts.to(torch.int32), tail_counts.to(torch.int32))


def select_with_plan(query, compressed_cache, block_table, query_start_loc, logical_positions, *, plan):
    """Record existing operators with fixed shape; admission is the caller's job.

    Padding IDs beyond group_counts are unspecified, exactly as in the native
    attention contract. Meaningful selected IDs and order are preserved.
    """
    if query.shape[0] != sum(plan.query_lens) or block_table.shape[0] != len(plan.query_lens):
        raise ValueError("captured query geometry does not match admission")
    if compressed_cache.shape[1] != plan.groups_per_block + 3 or block_table.shape[1] < plan.capacity_blocks:
        raise ValueError("captured cache/table geometry does not match admission")
    if logical_positions.shape != (query.shape[0],) or query_start_loc.numel() != len(plan.query_lens) + 1:
        raise ValueError("captured causal metadata shape does not match admission")
    if plan.backend == "prefix":
        return prefix_selection(
            logical_positions,
            compress_ratio=plan.compress_ratio,
            token_topk=plan.token_topk,
            capacity_groups=plan.capacity_groups,
        )
    current = qsa_indexer_select_groups_310(
        query,
        compressed_cache,
        block_table,
        query_start_loc,
        logical_positions,
        compress_ratio=plan.compress_ratio,
        token_topk=plan.token_topk,
        max_visible_groups=plan.capacity_groups,
        max_matmul_decode_tokens=plan.max_matmul_decode_tokens,
        query_lens=plan.query_lens,
        selection_policy=plan.selection_policy,
    )
    width = plan.token_topk // plan.compress_ratio
    if current.group_indices.shape[1] == width:
        return current
    destination = QSAGroupSelection(
        torch.empty((query.shape[0], width), dtype=torch.int32, device=query.device),
        torch.empty_like(current.group_counts),
        torch.empty_like(current.tail_starts),
        torch.empty_like(current.tail_counts),
    )
    copy_group_selection_into(destination, current)
    return destination
