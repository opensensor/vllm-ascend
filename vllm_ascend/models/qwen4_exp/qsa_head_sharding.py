# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel QSA query and KV head placement.

Every rank owns a disjoint range of query heads. Ranks sharing a KV head
replicate only that head; their row-parallel output projections are summed.
The indexer is deliberately replicated because its selection is shared by
all query heads and uses a separate cache.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QSAHeadShard:
    query_start: int
    num_query_heads: int
    kv_start: int
    num_kv_heads: int
    head_dim: int

    @property
    def query_rows(self) -> slice:
        start = self.query_start * self.head_dim
        return slice(start, start + self.num_query_heads * self.head_dim)

    @property
    def kv_rows(self) -> slice:
        start = self.kv_start * self.head_dim
        return slice(start, start + self.num_kv_heads * self.head_dim)


def qsa_head_shard(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> QSAHeadShard:
    """Return a GQA-aligned TP shard, rejecting unsupported head geometry."""
    if min(num_query_heads, num_kv_heads, head_dim, tp_size) <= 0:
        raise ValueError("QSA head counts, dimension, and TP size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"QSA TP rank {tp_rank} is outside TP size {tp_size}")
    if num_query_heads % num_kv_heads or num_query_heads % tp_size:
        raise ValueError("QSA query heads must divide evenly over KV heads and TP ranks")

    query_per_rank = num_query_heads // tp_size
    query_per_kv = num_query_heads // num_kv_heads
    query_start = tp_rank * query_per_rank
    if query_per_rank >= query_per_kv:
        if query_per_rank % query_per_kv:
            raise ValueError("QSA TP rank cannot own a fractional KV group")
        kv_start = query_start // query_per_kv
        kv_per_rank = query_per_rank // query_per_kv
    else:
        if query_per_kv % query_per_rank:
            raise ValueError("QSA TP ranks cannot split a KV group unevenly")
        kv_start = query_start // query_per_kv
        kv_per_rank = 1
    return QSAHeadShard(query_start, query_per_rank, kv_start, kv_per_rank, head_dim)
