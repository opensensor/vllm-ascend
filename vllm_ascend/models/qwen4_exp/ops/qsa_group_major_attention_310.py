# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental group-major QSA prefill for Ascend 310P.

The established prefill path gathers the selected K/V rows independently for
every query. Adjacent prefill queries often select overlapping compression
groups, so that layout can read and materialize the same K/V rows many times.
This module inverts the work inside a small query tile:

* form the sorted union of selected compression groups;
* gather each union group once from the paged NZ cache;
* run QK/PV against the union; and
* mask each query back to its original selected token set.

The implementation deliberately reuses the validated NZ gather operator. It
is an opt-in measurement candidate; the default serving path remains the
per-query batched gather until real-model NPU throughput and accuracy qualify
this schedule. A future fused kernel can consume the same group-major plan
without changing the model policy or numerical contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .qsa_gather_nz_310 import qsa_gather_key_transposed_nz_310, qsa_gather_value_nz_310
from .qsa_indexer import QSAGroupSelection
from .qsa_sparse_attention_310 import _validate_contract

QSA_GROUP_MAJOR_DEFAULT_QUERY_TILE = 8
QSA_GROUP_MAJOR_MAX_QUERY_TILE = 16
_NZ_INNER = 16


@dataclass(frozen=True)
class QSAGroupMajorPlan:
    """One request-local query tile expressed as unique groups and a mask."""

    group_indices: torch.Tensor
    token_mask: torch.Tensor
    device_group_count: torch.Tensor | None = None

    @property
    def unique_group_reads(self) -> int:
        if self.device_group_count is not None:
            raise RuntimeError("fixed-plan distinct count stays on device; use group_capacity for allocation")
        return self.group_indices.shape[0]

    @property
    def group_capacity(self) -> int:
        return self.group_indices.shape[0]


def _request_slices(query_lens: Sequence[int], num_tokens: int, num_requests: int) -> tuple[tuple[int, int], ...]:
    lengths = tuple(int(length) for length in query_lens)
    if len(lengths) != num_requests:
        raise ValueError("query_lens must contain one length per block-table row")
    if any(length < 0 for length in lengths):
        raise ValueError("query_lens must be nonnegative")
    if sum(lengths) != num_tokens:
        raise ValueError("query_lens must sum to the packed query token count")
    start = 0
    slices = []
    for length in lengths:
        stop = start + length
        slices.append((start, stop))
        start = stop
    return tuple(slices)


def build_qsa_group_major_plan(
    selection: QSAGroupSelection,
    *,
    compress_ratio: int = 4,
) -> QSAGroupMajorPlan:
    """Build a device-resident union and exact per-query token mask.

    ``torch.unique`` and ``torch.searchsorted`` remain on the input device.
    No selected indices are copied to the host. The finite causal tail is
    represented as a partially enabled compression group, allowing the same
    union gather to serve both learned groups and tail tokens.
    """
    if selection.group_indices.ndim != 2:
        raise ValueError("group_indices must be [T,K]")
    num_queries, group_width = selection.group_indices.shape
    if num_queries <= 0 or group_width <= 0:
        raise ValueError("group-major planning requires a nonempty selection")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    for name, tensor in (
        ("group_counts", selection.group_counts),
        ("tail_starts", selection.tail_starts),
        ("tail_counts", selection.tail_counts),
    ):
        if tensor.shape != (num_queries,):
            raise ValueError(f"{name} must be [T]")
        if tensor.device != selection.group_indices.device:
            raise ValueError(f"{name} must be on the group_indices device")

    device = selection.group_indices.device
    group_indices = selection.group_indices.to(torch.int64)
    group_counts = selection.group_counts.to(torch.int64)
    tail_starts = selection.tail_starts.to(torch.int64)
    tail_counts = selection.tail_counts.to(torch.int64)
    group_ranks = torch.arange(group_width, dtype=torch.int64, device=device)
    group_valid = group_ranks.unsqueeze(0) < group_counts.unsqueeze(1)
    tail_valid = tail_counts > 0
    tail_groups = torch.div(tail_starts, compress_ratio, rounding_mode="floor")

    candidates = torch.cat((group_indices.reshape(-1), tail_groups), dim=0)
    candidate_valid = torch.cat((group_valid.reshape(-1), tail_valid), dim=0)
    valid_candidates = candidates[candidate_valid]
    if valid_candidates.numel() == 0:
        raise ValueError("group-major planning requires at least one selected group or tail token")
    union_groups = torch.unique(valid_candidates, sorted=True)

    # Map every fixed-width selection entry into the union. Invalid padded
    # entries use column zero and scatter a zero, so they cannot enable a token.
    group_columns = torch.searchsorted(union_groups, group_indices.clamp_min(0))
    padding_column = union_groups.shape[0]
    group_columns = torch.where(group_valid, group_columns, torch.full_like(group_columns, padding_column))
    group_membership = torch.zeros(
        (num_queries, padding_column + 1),
        dtype=torch.bool,
        device=device,
    )
    group_membership.scatter_(1, group_columns, group_valid)
    token_mask = group_membership[:, :padding_column].unsqueeze(-1).expand(-1, -1, compress_ratio).clone()

    # The tail starts on a compression-group boundary and enables only the
    # causal prefix of that group. If a selector ever also returns the same
    # group, OR preserves the complete learned selection.
    tail_columns = torch.searchsorted(union_groups, tail_groups.clamp_min(0))
    union_columns = torch.arange(union_groups.shape[0], dtype=torch.int64, device=device)
    tail_group_mask = union_columns.unsqueeze(0) == tail_columns.unsqueeze(1)
    tail_offsets = torch.arange(compress_ratio, dtype=torch.int64, device=device)
    tail_token_mask = (
        tail_group_mask.unsqueeze(-1)
        & tail_valid[:, None, None]
        & (tail_offsets[None, None, :] < tail_counts[:, None, None])
    )
    token_mask.logical_or_(tail_token_mask)
    return QSAGroupMajorPlan(
        group_indices=union_groups.to(torch.int32),
        token_mask=token_mask.reshape(num_queries, -1),
    )


def build_qsa_fixed_group_major_plan(
    selection: QSAGroupSelection,
    *,
    compress_ratio: int = 4,
) -> QSAGroupMajorPlan:
    """Fixed-shape union without unique, nonzero or masked_select.

    Capacity is the host-known maximum number of selected groups plus tails.
    Only distinct valid groups are read from KV; unused output lanes remain
    allocated and are masked out. QK/PV work can increase at poor overlap, so
    this schedule is an opt-in candidate rather than the prefill default.
    """
    groups = selection.group_indices
    if groups.ndim != 2 or min(groups.shape) <= 0 or compress_ratio <= 0:
        raise ValueError("fixed group-major planning needs nonempty [T,K] groups and positive ratio")
    queries, width = groups.shape
    for tensor in (selection.group_counts, selection.tail_starts, selection.tail_counts):
        if tensor.shape != (queries,) or tensor.device != groups.device:
            raise ValueError("selection vectors must be [T] on the group device")
    group_valid = torch.arange(width, device=groups.device)[None, :] < selection.group_counts[:, None]
    tail_valid = selection.tail_counts > 0
    tail_groups = torch.div(selection.tail_starts, compress_ratio, rounding_mode="floor")
    candidates = torch.cat((groups.reshape(-1), tail_groups)).to(torch.int32)
    candidate_valid = torch.cat((group_valid.reshape(-1), tail_valid))
    sentinel = torch.iinfo(torch.int32).max
    candidates = torch.where(candidate_valid, candidates, sentinel)
    ordered, order = torch.sort(candidates, stable=True)
    unique = torch.cat((torch.ones_like(ordered[:1], dtype=torch.bool), ordered[1:] != ordered[:-1]))
    unique = unique & (ordered != sentinel)
    compact_columns = (unique.to(torch.int64).cumsum(0) - 1).clamp_min(0)
    capacity = candidates.numel()
    union = torch.zeros_like(candidates)
    union.scatter_add_(0, compact_columns, torch.where(unique, ordered, 0))
    original_columns = torch.empty_like(order)
    original_columns.scatter_(0, order, compact_columns)
    group_columns = original_columns[: queries * width].reshape(queries, width)
    group_columns = torch.where(group_valid, group_columns, capacity)
    membership = torch.zeros((queries, capacity + 1), dtype=torch.bool, device=groups.device)
    membership.scatter_(1, group_columns, group_valid)
    mask = membership[:, :capacity, None].expand(-1, -1, compress_ratio).clone()
    tail_columns = original_columns[queries * width :]
    offsets = torch.arange(compress_ratio, device=groups.device)
    mask.logical_or_(
        (torch.arange(capacity, device=groups.device)[None, :, None] == tail_columns[:, None, None])
        & tail_valid[:, None, None]
        & (offsets[None, None, :] < selection.tail_counts[:, None, None])
    )
    return QSAGroupMajorPlan(union, mask.reshape(queries, -1), unique.sum().reshape(1).to(torch.int32))


def _group_major_attention(
    query: torch.Tensor,
    selected_keys: torch.Tensor,
    selected_values: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Apply GQA attention to one gathered union on CPU or NPU."""
    num_queries, num_query_heads, head_dim = query.shape
    if selected_keys.ndim != 4 or selected_values.ndim != 4:
        raise ValueError("selected K/V must be [1,H,D,S] and [1,H,S,D]")
    if selected_keys.shape[0] != 1 or selected_values.shape[0] != 1:
        raise ValueError("group-major K/V must contain one union row")
    num_kv_heads = selected_keys.shape[1]
    selected_tokens = selected_keys.shape[-1]
    if selected_keys.shape[2] != head_dim:
        raise ValueError("selected key head dimension does not match query")
    if selected_values.shape != (1, num_kv_heads, selected_tokens, head_dim):
        raise ValueError("selected value shape does not match selected keys")
    if token_mask.shape != (num_queries, selected_tokens):
        raise ValueError("token_mask must match query and selected-token dimensions")
    if (
        selected_keys.device != query.device
        or selected_values.device != query.device
        or token_mask.device != query.device
    ):
        raise ValueError("query, selected K/V, and token_mask must share a device")
    if selected_keys.dtype != query.dtype or selected_values.dtype != query.dtype:
        raise ValueError("query and selected K/V must share a dtype")
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    heads_per_kv_head = num_query_heads // num_kv_heads
    # Make the shared union a real batch of two KV-head matrices. Flattening
    # query rows within each KV head avoids broadcasting the same K/V tensor
    # across the query-tile dimension, which selects a much slower 310P
    # BatchMatMul schedule. Keeping logits head-major also lets PV consume the
    # softmax result without a large transpose/copy.
    query_rows = (
        query.view(num_queries, num_kv_heads, heads_per_kv_head, head_dim)
        .permute(1, 0, 2, 3)
        .reshape(num_kv_heads, num_queries * heads_per_kv_head, head_dim)
    )
    prescale_query = scale > 0 and math.frexp(scale)[0] == 0.5
    score_query = query_rows * scale if prescale_query else query_rows
    logits = torch.matmul(score_query, selected_keys[0]).view(
        num_kv_heads,
        num_queries,
        heads_per_kv_head,
        selected_tokens,
    )
    logits = (logits.float() if prescale_query else logits.float() * scale).masked_fill(
        ~token_mask[None, :, None, :], -torch.inf
    )
    probabilities = torch.softmax(logits, dim=-1).to(query.dtype)
    # Fixed-shape plans can contain inactive rows. Preserve NaN propagation
    # for active input rows while giving fully masked placeholders zero output.
    probabilities = torch.where(token_mask.any(dim=-1)[None, :, None, None], probabilities, 0)
    output = torch.matmul(
        probabilities.view(num_kv_heads, num_queries * heads_per_kv_head, selected_tokens),
        selected_values[0],
    ).view(num_kv_heads, num_queries, heads_per_kv_head, head_dim)
    return output.permute(1, 0, 2, 3).reshape(num_queries, num_query_heads, head_dim)


def qsa_group_major_prefill_310(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    scale: float,
    compress_ratio: int = 4,
    query_lens: Sequence[int] | None = None,
    query_tile: int = QSA_GROUP_MAJOR_DEFAULT_QUERY_TILE,
    fixed_plan: bool = False,
) -> torch.Tensor:
    """Run the opt-in union-gather QSA prefill schedule on 310P."""
    _validate_contract(query, key_cache, value_cache, selection, block_table, query_start_loc, compress_ratio)
    if query.device.type != "npu":
        raise RuntimeError("group-major 310P QSA prefill requires an Ascend NPU")
    if query_tile < 1 or query_tile > QSA_GROUP_MAJOR_MAX_QUERY_TILE:
        raise ValueError(f"query_tile must be in [1, {QSA_GROUP_MAJOR_MAX_QUERY_TILE}]")

    num_tokens, _, head_dim = query.shape
    if query_lens is None:
        if block_table.shape[0] != 1:
            raise ValueError("multi-request group-major QSA requires host query_lens")
        request_slices = ((0, num_tokens),)
    else:
        request_slices = _request_slices(query_lens, num_tokens, block_table.shape[0])

    outputs = []
    for request, (request_start, request_stop) in enumerate(request_slices):
        request_table = block_table[request : request + 1]
        for start in range(request_start, request_stop, query_tile):
            stop = min(start + query_tile, request_stop)
            tile_selection = QSAGroupSelection(
                group_indices=selection.group_indices[start:stop],
                group_counts=selection.group_counts[start:stop],
                tail_starts=selection.tail_starts[start:stop],
                tail_counts=selection.tail_counts[start:stop],
            )
            planner = build_qsa_fixed_group_major_plan if fixed_plan else build_qsa_group_major_plan
            plan = planner(tile_selection, compress_ratio=compress_ratio)
            union_size = plan.group_indices.shape[0]
            union_selection = QSAGroupSelection(
                group_indices=plan.group_indices.unsqueeze(0),
                group_counts=(
                    plan.device_group_count
                    if plan.device_group_count is not None
                    else torch.full((1,), union_size, dtype=torch.int32, device=query.device)
                ),
                tail_starts=torch.zeros(1, dtype=torch.int32, device=query.device),
                tail_counts=torch.zeros(1, dtype=torch.int32, device=query.device),
            )
            selected_keys = qsa_gather_key_transposed_nz_310(
                key_cache,
                union_selection,
                request_table,
                head_dim=head_dim,
            )
            selected_values = qsa_gather_value_nz_310(
                value_cache,
                union_selection,
                request_table,
                head_dim=head_dim,
            )
            padded_tokens = selected_keys.shape[-1]
            token_mask = torch.zeros(
                (stop - start, padded_tokens),
                dtype=torch.bool,
                device=query.device,
            )
            token_mask[:, : plan.token_mask.shape[1]].copy_(plan.token_mask)
            outputs.append(
                _group_major_attention(
                    query[start:stop],
                    selected_keys,
                    selected_values,
                    token_mask,
                    scale=scale,
                )
            )
    return torch.cat(outputs, dim=0)


__all__ = [
    "QSA_GROUP_MAJOR_DEFAULT_QUERY_TILE",
    "QSA_GROUP_MAJOR_MAX_QUERY_TILE",
    "QSAGroupMajorPlan",
    "build_qsa_group_major_plan",
    "build_qsa_fixed_group_major_plan",
    "qsa_group_major_prefill_310",
]
