# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-process gloo runtime for the QSA-aware DCP prototype (plan T8.2).

Each process is one decode-context-parallel rank. The main QSA K/V cache is
sequence-sharded (:mod:`.sharding`) so a rank holds only its stride of rows; the
indexer selection is replicated, so every rank derives the same selection. The
ranks then exchange ONLY the indexer-selected rows (a per-rank
``all_gather_object`` of the owned K/V blocks -- never the whole cache) and each
rank rebuilds every owner's partial and folds them in fixed rank order
(:func:`.attention.merge_partials`) into the final gated output.

The worker is intentionally small: it plumbs the ``gloo`` collectives and reuses
the pure-host core in :mod:`.attention`, so the multi-process result is the same
math the in-process simulation (:func:`.attention.qsa_dcp_sparse_attention`)
already matches against the T6.2/T6.4 single-rank reference.

Host only (CPU ``gloo``); no NPU / Triton import. PROTOTYPE for measurement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .attention import finalize_output, merge_partials, partial_from_owned_rows
from .sharding import QSAShardPlan
from .transfer import row_bytes


@dataclass(frozen=True)
class DCPWorkerConfig:
    """Inputs a gloo worker needs (identical, replicated, on every rank)."""

    num_ranks: int
    num_kv_heads: int
    scale: float
    accum_dtype: torch.dtype
    master_addr: str = "127.0.0.1"
    master_port: str = "29591"


def _owned_rows_per_token(
    shard: torch.Tensor,
    plan: QSAShardPlan,
    rank: int,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
) -> list[torch.Tensor]:
    """This rank's owned selected rows, one ``[m_t, Hkv, D]`` block per query."""
    blocks: list[torch.Tensor] = []
    for token in range(packed_indices.shape[0]):
        count = max(int(valid_counts[token].item()), 0)
        idx = packed_indices[token, :count]
        idx = idx[idx >= 0].to(torch.long)
        owned = idx[idx % plan.num_ranks == rank]
        local_slots = owned // plan.num_ranks
        blocks.append(shard.index_select(0, local_slots))
    return blocks


def run_dcp_rank(
    rank: int,
    config: DCPWorkerConfig,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Run one DCP rank end to end on an initialised ``gloo`` process group.

    The full ``key_cache`` / ``value_cache`` are passed in for a self-contained
    host test; the rank immediately shards them and only ever touches its own
    stride plus the selected rows exchanged from peers.

    Returns:
        ``(out, selected_rows_moved)`` -- the gated ``[T, Hq, D]`` output (identical
        on every rank) and the total selected rows that crossed the exchange.
    """
    plan = QSAShardPlan(num_ranks=config.num_ranks)
    key_shard = plan.split_cache(key_cache)[rank]
    value_shard = plan.split_cache(value_cache)[rank]

    my_key_blocks = _owned_rows_per_token(key_shard, plan, rank, packed_indices, valid_counts)
    my_value_blocks = _owned_rows_per_token(value_shard, plan, rank, packed_indices, valid_counts)

    # Exchange ONLY the selected rows: gather every rank's owned K/V blocks.
    gathered_keys: list[list[torch.Tensor]] = [[] for _ in range(config.num_ranks)]
    gathered_values: list[list[torch.Tensor]] = [[] for _ in range(config.num_ranks)]
    dist.all_gather_object(gathered_keys, my_key_blocks)
    dist.all_gather_object(gathered_values, my_value_blocks)

    # Deterministic reduction: rebuild each owner's partial and fold rank 0..R-1.
    partials = [
        partial_from_owned_rows(
            query,
            gathered_keys[owner],
            gathered_values[owner],
            num_kv_heads=config.num_kv_heads,
            scale=config.scale,
            accum_dtype=config.accum_dtype,
        )
        for owner in range(config.num_ranks)
    ]
    merged = merge_partials(partials)
    out = finalize_output(
        merged,
        gate,
        num_query_heads=query.shape[1],
        head_dim=query.shape[-1],
        accum_dtype=config.accum_dtype,
    )

    my_rows = sum(block.shape[0] for block in my_key_blocks)
    total_rows = torch.tensor(my_rows, dtype=torch.int64)
    dist.all_reduce(total_rows, op=dist.ReduceOp.SUM)
    return out, int(total_rows.item())


def _worker_entry(rank: int, config: DCPWorkerConfig, payload: dict, result_dir: str) -> None:
    """Process entry point: init gloo, run the rank, save the result, tear down."""
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    dist.init_process_group(backend="gloo", rank=rank, world_size=config.num_ranks)
    try:
        out, total_rows = run_dcp_rank(
            rank,
            config,
            payload["query"],
            payload["key_cache"],
            payload["value_cache"],
            payload["gate"],
            payload["packed_indices"],
            payload["valid_counts"],
        )
        torch.save(
            {
                "out": out,
                "total_rows": total_rows,
                "row_bytes": row_bytes(config.num_kv_heads, out.shape[-1], payload["key_cache"].dtype),
            },
            os.path.join(result_dir, f"rank_{rank}.pt"),
        )
    finally:
        dist.barrier()
        dist.destroy_process_group()


__all__ = ["DCPWorkerConfig", "run_dcp_rank", "_worker_entry"]
