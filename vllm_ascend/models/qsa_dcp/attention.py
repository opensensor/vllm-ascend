# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA-aware Decode Context Parallel attention core (plan T8.2, Candidate B).

The single-rank ground truth is the T6.2 gather-based sparse GQA attention
(:func:`vllm_ascend.models.qwen4_exp.ops.qsa_attention.qsa_sparse_gqa_attention`),
which the T6.4 decoder path composes. This module reproduces that math with the
main K/V cache *sequence-sharded* across ranks:

1. The indexer history is REPLICATED, so every rank derives the identical
   ``packed_indices`` / ``valid_counts`` selection (deterministic top-k upstream).
2. Each rank owns a round-robin K/V shard (:mod:`.sharding`). For a query it
   gathers ONLY the selected rows it owns and computes a local partial attention
   state ``(m, l, acc)`` -- running max, running ``sum exp``, and the running
   weighted value sum, all in the accumulation dtype.
3. The per-rank partials are combined by a FIXED-order online-softmax reduction
   (rank 0, 1, 2, 3) -- a flash-attention merge that is mathematically exact and,
   because the fold order is pinned, bitwise stable across reruns.
4. The reduced context is divided by the global normaliser and gated with
   ``* sigmoid(gate)`` (zero-selection queries stay zero, matching T6.2).

Only the selected rows are ever gathered; the whole cache is never all-gathered.
The exchange volume is recorded by :class:`.transfer.TransferLedger`.

Pure torch, host only: no NPU / Triton import. This is a PROTOTYPE for host
measurement; it pre-decides nothing (D4 decides on hardware).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sharding import QSAShardPlan
from .transfer import TransferLedger, row_bytes

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class QSAPartialState:
    """One rank's local partial attention state for every query token.

    Shapes (``T`` queries, ``Hkv`` kv heads, ``g`` query heads per kv head,
    ``D`` head dim):

    * ``max_score``: ``[T, Hkv, g]`` running max logit (``-inf`` if the rank owns
      no selected row for that query/head).
    * ``sum_exp``: ``[T, Hkv, g]`` running ``sum(exp(score - max))``.
    * ``weighted_value``: ``[T, Hkv, g, D]`` running ``sum(exp(...) * value)``.
    """

    max_score: torch.Tensor
    sum_exp: torch.Tensor
    weighted_value: torch.Tensor


def _empty_partial(
    num_tokens: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> QSAPartialState:
    return QSAPartialState(
        max_score=torch.full((num_tokens, num_kv_heads, group_size), _NEG_INF, dtype=dtype, device=device),
        sum_exp=torch.zeros((num_tokens, num_kv_heads, group_size), dtype=dtype, device=device),
        weighted_value=torch.zeros((num_tokens, num_kv_heads, group_size, head_dim), dtype=dtype, device=device),
    )


def _valid_selected_positions(packed_row: torch.Tensor, valid_count: int) -> torch.Tensor:
    """First ``valid_count`` non-``-1`` global positions of one packed row."""
    count = max(int(valid_count), 0)
    idx = packed_row[:count]
    return idx[idx >= 0].to(torch.long)


def _accumulate_token_partial(
    partial: QSAPartialState,
    token: int,
    q_grp: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    scale: float,
) -> None:
    """Fill ``partial`` at ``token`` from one query's gathered K/V rows.

    Args:
        partial: partial state to write into (modified in place).
        token: query index being filled.
        q_grp: ``[Hkv, g, D]`` grouped query for this token.
        k_rows, v_rows: ``[m, Hkv, D]`` gathered selected rows (``m >= 1``).
        scale: softmax scale.
    """
    scores = torch.einsum("kgd,mkd->kgm", q_grp, k_rows) * scale
    local_max = scores.amax(dim=-1)  # [Hkv, g]
    exp_scores = torch.exp(scores - local_max.unsqueeze(-1))
    partial.max_score[token] = local_max
    partial.sum_exp[token] = exp_scores.sum(dim=-1)
    partial.weighted_value[token] = torch.einsum("kgm,mkd->kgd", exp_scores, v_rows)


def compute_rank_partial(
    query: torch.Tensor,
    key_shard: torch.Tensor,
    value_shard: torch.Tensor,
    *,
    plan: QSAShardPlan,
    rank: int,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    accum_dtype: torch.dtype,
) -> tuple[QSAPartialState, int]:
    """Local partial attention for ``rank`` over the selected rows it owns.

    Args:
        query: ``[T, Hq, D]`` queries (already normed / RoPE'd upstream).
        key_shard, value_shard: ``[n_rank, Hkv, D]`` this rank's round-robin
            shard of the main K/V cache.
        plan: the sequence-shard map.
        rank: this rank's index.
        packed_indices: ``[T, W]`` replicated selection (``-1`` padded).
        valid_counts: ``[T]`` per-query valid-entry count.
        num_kv_heads: KV head count for GQA grouping.
        scale: softmax scale (``head_dim ** -0.5``).
        accum_dtype: score / exp / weighted-sum dtype.

    Returns:
        ``(partial, owned_rows)`` -- the rank's partial state and the number of
        selected rows it gathered from its shard (the exchange volume it emits).
    """
    if query.ndim != 3:
        raise ValueError("query must be [T, Hq, D]")
    num_tokens, num_q_heads, head_dim = query.shape
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by kv heads")
    group_size = num_q_heads // num_kv_heads

    q = query.to(accum_dtype)
    k_shard = key_shard.to(accum_dtype)
    v_shard = value_shard.to(accum_dtype)
    partial = _empty_partial(num_tokens, num_kv_heads, group_size, head_dim, accum_dtype, query.device)
    owned_rows = 0

    for token in range(num_tokens):
        positions = _valid_selected_positions(packed_indices[token], int(valid_counts[token].item()))
        if positions.numel() == 0:
            continue
        local_slots = plan.owned_local_slots(positions, rank)
        if local_slots.numel() == 0:
            continue
        owned_rows += int(local_slots.numel())
        k_local = k_shard.index_select(0, local_slots)  # [m, Hkv, D]
        v_local = v_shard.index_select(0, local_slots)  # [m, Hkv, D]
        q_grp = q[token].reshape(num_kv_heads, group_size, head_dim)
        _accumulate_token_partial(partial, token, q_grp, k_local, v_local, scale)

    return partial, owned_rows


def partial_from_owned_rows(
    query: torch.Tensor,
    owned_key_rows: list[torch.Tensor],
    owned_value_rows: list[torch.Tensor],
    *,
    num_kv_heads: int,
    scale: float,
    accum_dtype: torch.dtype,
) -> QSAPartialState:
    """Build a rank's partial state from already-exchanged selected rows.

    The multi-process gloo path exchanges the selected rows themselves (one K and
    one V tensor per query token), then every rank rebuilds each owner's partial
    from those rows with this function before the fixed-order merge. It mirrors
    :func:`compute_rank_partial` but reads explicit row blocks instead of gathering
    from a local shard.

    Args:
        query: ``[T, Hq, D]`` normed / RoPE'd queries.
        owned_key_rows, owned_value_rows: length-``T`` lists of ``[m_t, Hkv, D]``
            selected rows this owner contributed for each query token.
        num_kv_heads: KV head count for GQA grouping.
        scale: softmax scale.
        accum_dtype: accumulation dtype.
    """
    num_tokens, num_q_heads, head_dim = query.shape
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by kv heads")
    group_size = num_q_heads // num_kv_heads
    q = query.to(accum_dtype)
    partial = _empty_partial(num_tokens, num_kv_heads, group_size, head_dim, accum_dtype, query.device)
    for token in range(num_tokens):
        k_rows = owned_key_rows[token]
        if k_rows.shape[0] == 0:
            continue
        q_grp = q[token].reshape(num_kv_heads, group_size, head_dim)
        _accumulate_token_partial(
            partial,
            token,
            q_grp,
            k_rows.to(accum_dtype),
            owned_value_rows[token].to(accum_dtype),
            scale,
        )
    return partial


def merge_partials(partials: list[QSAPartialState]) -> QSAPartialState:
    """Deterministic online-softmax merge of per-rank partials in list order.

    The fold visits ``partials`` left to right (the caller passes them in rank
    order 0, 1, 2, 3), so the reduction order is fixed and the result is bitwise
    stable across reruns. The merge is the exact flash-attention combine, so the
    reduced state equals a single softmax over the union of all ranks' rows.
    """
    if not partials:
        raise ValueError("merge_partials needs at least one partial state")

    running = partials[0]
    for other in partials[1:]:
        new_max = torch.maximum(running.max_score, other.max_score)
        # exp(-inf - -inf) is NaN; a rank that contributed nothing (max == -inf)
        # must scale by 0, so scrub the NaN back to 0.
        alpha = torch.exp(running.max_score - new_max)
        beta = torch.exp(other.max_score - new_max)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        beta = torch.nan_to_num(beta, nan=0.0)
        sum_exp = running.sum_exp * alpha + other.sum_exp * beta
        weighted = running.weighted_value * alpha.unsqueeze(-1) + other.weighted_value * beta.unsqueeze(-1)
        running = QSAPartialState(max_score=new_max, sum_exp=sum_exp, weighted_value=weighted)
    return running


def finalize_output(
    merged: QSAPartialState,
    gate: torch.Tensor,
    *,
    num_query_heads: int,
    head_dim: int,
    accum_dtype: torch.dtype,
    apply_output_gate: bool = True,
) -> torch.Tensor:
    """Normalise the merged state and apply the ``* sigmoid(gate)`` output gate.

    Queries whose global ``sum_exp`` is zero (no rank selected anything) stay
    zero, matching the T6.2 zero-selection semantics.
    """
    num_tokens = merged.sum_exp.shape[0]
    has_rows = merged.sum_exp > 0
    safe_denominator = torch.where(has_rows, merged.sum_exp, torch.ones_like(merged.sum_exp))
    context = merged.weighted_value / safe_denominator.unsqueeze(-1)
    context = torch.where(has_rows.unsqueeze(-1), context, torch.zeros_like(context))
    out = context.reshape(num_tokens, num_query_heads, head_dim)
    if apply_output_gate:
        out = out * torch.sigmoid(gate.to(accum_dtype))
    return out


def qsa_dcp_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    num_kv_heads: int,
    *,
    num_ranks: int,
    plan: QSAShardPlan | None = None,
    scale: float | None = None,
    accum_dtype: torch.dtype = torch.float32,
    apply_output_gate: bool = True,
) -> tuple[torch.Tensor, TransferLedger]:
    """In-process ``num_ranks``-way DCP simulation of QSA sparse attention.

    Shards ``key_cache`` / ``value_cache`` across ranks, computes each rank's
    partial over its owned selected rows, merges the partials in fixed rank
    order, and finalises -- the exact host analogue of the multi-process gloo
    path, returning the same result plus the selected-row transfer ledger.

    Args:
        query: ``[T, Hq, D]`` normed / RoPE'd queries.
        key_cache, value_cache: ``[S, Hkv, D]`` full-context caches (sharded here).
        gate: ``[T, Hq, D]`` pre-sigmoid output gate.
        packed_indices: ``[T, W]`` selection (``-1`` padded), replicated on all ranks.
        valid_counts: ``[T]`` per-query valid-entry count.
        num_kv_heads: KV head count for GQA grouping.
        num_ranks: DCP world size (4 for the 310P box).
        plan: shard map; defaults to a round-robin plan for ``num_ranks``.
        scale: softmax scale; defaults to ``head_dim ** -0.5``.
        accum_dtype: accumulation dtype.
        apply_output_gate: apply ``* sigmoid(gate)`` when ``True``.

    Returns:
        ``(out, ledger)`` -- ``[T, Hq, D]`` gated output in ``accum_dtype`` and the
        :class:`TransferLedger` proving only selected rows moved.
    """
    if key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("key/value caches must be [S, Hkv, D]")
    head_dim = query.shape[-1]
    shard_plan = plan or QSAShardPlan(num_ranks=num_ranks)
    if shard_plan.num_ranks != num_ranks:
        raise ValueError("plan.num_ranks must match num_ranks")
    softmax_scale = head_dim**-0.5 if scale is None else float(scale)

    key_shards = shard_plan.split_cache(key_cache)
    value_shards = shard_plan.split_cache(value_cache)
    ledger = TransferLedger(row_bytes=row_bytes(num_kv_heads, head_dim, key_cache.dtype))

    partials: list[QSAPartialState] = []
    for rank in range(num_ranks):
        partial, owned_rows = compute_rank_partial(
            query,
            key_shards[rank],
            value_shards[rank],
            plan=shard_plan,
            rank=rank,
            packed_indices=packed_indices,
            valid_counts=valid_counts,
            num_kv_heads=num_kv_heads,
            scale=softmax_scale,
            accum_dtype=accum_dtype,
        )
        partials.append(partial)
        ledger.record_selected_rows(owned_rows)

    merged = merge_partials(partials)
    out = finalize_output(
        merged,
        gate,
        num_query_heads=query.shape[1],
        head_dim=head_dim,
        accum_dtype=accum_dtype,
        apply_output_gate=apply_output_gate,
    )
    return out, ledger


__all__ = [
    "QSAPartialState",
    "compute_rank_partial",
    "partial_from_owned_rows",
    "merge_partials",
    "finalize_output",
    "qsa_dcp_sparse_attention",
]
