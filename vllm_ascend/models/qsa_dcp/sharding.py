# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-sharding map for the QSA-aware Decode Context Parallel prototype (T8.2).

Candidate B of the 1M-cache decision shards the *main* QSA K/V cache across the
four Ascend 310P ranks along the sequence (position) axis, while the indexer
history stays fully replicated on every rank. This module owns the
position <-> (owner rank, local slot) map only; it holds no tensors and imports
nothing NPU-specific, so the whole sharding contract is host-testable.

The map is the interleave-1 round-robin used by vLLM-Ascend's decode context
parallel (see ``attention/context_parallel/common_cp.get_dcp_local_seq_lens``
with ``interleave_size == 1``): global position ``p`` is owned by rank
``p % num_ranks`` at local slot ``p // num_ranks``. Round-robin (rather than
contiguous blocks) keeps the per-rank shard lengths balanced to within one row
for any context length, which the T0.5 placement-imbalance check requires.

This is a PROTOTYPE for host measurement. It pre-decides nothing; D4 decides on
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Interleave granularity of the round-robin shard map. The prototype pins it to 1
# (pure round-robin) so every rank owns a maximally balanced, evenly strided set
# of positions; the vLLM DCP helper generalises this to larger interleave tiles.
DEFAULT_INTERLEAVE_SIZE = 1


@dataclass(frozen=True)
class QSAShardPlan:
    """Round-robin sequence-shard map for the main QSA K/V cache.

    Attributes:
        num_ranks: number of decode-context-parallel ranks (4 for the 310P box).
        interleave_size: round-robin tile width (pinned to 1 for the prototype).
    """

    num_ranks: int
    interleave_size: int = DEFAULT_INTERLEAVE_SIZE

    def __post_init__(self) -> None:
        if self.num_ranks < 1:
            raise ValueError("num_ranks must be >= 1")
        if self.interleave_size != DEFAULT_INTERLEAVE_SIZE:
            raise NotImplementedError("the T8.2 prototype pins interleave_size to 1")

    def owner_of(self, position: int) -> int:
        """Rank that owns global ``position``."""
        return int(position) % self.num_ranks

    def local_slot_of(self, position: int) -> int:
        """Local slot of global ``position`` within its owner's shard."""
        return int(position) // self.num_ranks

    def shard_length(self, total_len: int, rank: int) -> int:
        """Number of positions in ``[0, total_len)`` owned by ``rank``."""
        if not 0 <= rank < self.num_ranks:
            raise ValueError(f"rank {rank} out of range for {self.num_ranks} ranks")
        base = total_len // self.num_ranks
        remainder = total_len - base * self.num_ranks
        return base + (1 if rank < remainder else 0)

    def split_cache(self, cache: torch.Tensor) -> list[torch.Tensor]:
        """Split a ``[S, ...]`` cache into ``num_ranks`` round-robin shards.

        Shard ``r`` is ``cache[r::num_ranks]``; local slot ``j`` of shard ``r``
        is global position ``r + j * num_ranks``.
        """
        return [cache[rank :: self.num_ranks] for rank in range(self.num_ranks)]

    def map_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global ``positions`` to ``(owner_ranks, local_slots)`` tensors.

        Negative entries (``-1`` selection padding) pass through unchanged in both
        outputs so callers can mask them out.
        """
        pos = positions.to(torch.long)
        valid = pos >= 0
        owners = torch.where(valid, pos % self.num_ranks, pos)
        local = torch.where(valid, pos // self.num_ranks, pos)
        return owners, local

    def owned_local_slots(self, positions: torch.Tensor, rank: int) -> torch.Tensor:
        """Local slots (in ``rank``'s shard) of the ``positions`` owned by ``rank``.

        ``positions`` is a 1-D tensor of valid (``>= 0``) global positions; the
        result preserves their order, so a downstream gather is deterministic.
        """
        pos = positions.to(torch.long)
        owned = pos[pos % self.num_ranks == rank]
        return owned // self.num_ranks


__all__ = ["DEFAULT_INTERLEAVE_SIZE", "QSAShardPlan"]
