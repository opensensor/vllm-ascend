# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank memory accounting for the Qwen3.8-Flash-Next (Qwen4Exp) 1M path.

This harness records device- and host-resident bytes by component during weight
load and cache allocation for the four-chip Ascend 310P deployment, then reports a
machine-readable breakdown (consumed by the run-log observability, plan TOBS) and a
human summary. It fails startup when per-rank *device* placement is imbalanced
beyond a fixed threshold without an explicit approved reason.

The module is pure Python (no torch / torch-npu import) so it is fully host
testable with synthetic allocation traces. Real device statistics are injected by
the caller; the harness never queries the accelerator itself.

Byte figures are decimal-neutral: callers pass raw byte counts and the human
summary renders GiB (1024**3). Component definitions mirror
``docs/source/developer_guide/Design_Documents/qwen38_flash_next_1m_runtime_requirements.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

# Per-rank device placement imbalance above this fraction fails startup unless an
# approved reason is supplied. Balanced TP4/EP4 placement is a hard requirement
# for the capacity model (PRD R2/R3, runtime-requirements doc §4/§5).
MAX_PLACEMENT_IMBALANCE = 0.05

_BYTES_PER_GIB = 1024**3


class MemComponent(str, Enum):
    """Memory components tracked per rank.

    Membership in :data:`HOST_COMPONENTS` distinguishes device-resident components
    (counted toward the per-chip NPU budget and the imbalance check) from
    host-resident ones (the shared PLE table, which is not sharded and must not be
    counted per rank).
    """

    EMBEDDING = "embedding"
    NON_EXPERT_FP16 = "non_expert_fp16"
    EXPERT_W8A8 = "expert_w8a8"
    QUANT_SCALES = "quant_scales"
    QSA_MAIN_KV = "qsa_main_kv"
    INDEXER_RING = "indexer_ring"
    INDEXER_COMPRESSED = "indexer_compressed"
    GDN_STATE = "gdn_state"
    WORKSPACES = "workspaces"
    PLE_HOST_TABLE = "ple_host_table"


# Components that live in host RAM (shared across ranks), excluded from per-rank
# device totals and from the placement-imbalance check.
HOST_COMPONENTS: frozenset[MemComponent] = frozenset({MemComponent.PLE_HOST_TABLE})


class PlacementImbalanceError(RuntimeError):
    """Raised when per-rank device bytes are imbalanced beyond the threshold."""


@dataclass
class RankMemoryReport:
    """Accumulates memory bytes for a single rank.

    ``free_bytes`` and ``peak_bytes`` are optional device totals supplied by the
    caller from real allocator statistics; they are reported but not part of the
    component sum.
    """

    rank: int
    components: dict[MemComponent, int] = field(default_factory=dict)
    free_bytes: int | None = None
    peak_bytes: int | None = None

    def add(self, component: MemComponent, num_bytes: int) -> None:
        """Record ``num_bytes`` for ``component`` on this rank (accumulates)."""
        if num_bytes < 0:
            raise ValueError(f"num_bytes for {component.value} must be >= 0, got {num_bytes}")
        self.components[component] = self.components.get(component, 0) + num_bytes

    def set_device_totals(self, *, free_bytes: int, peak_bytes: int) -> None:
        """Record allocator-reported free and peak device bytes for this rank."""
        self.free_bytes = free_bytes
        self.peak_bytes = peak_bytes

    def device_bytes(self) -> int:
        """Sum of device-resident component bytes (excludes host components)."""
        return sum(num_bytes for component, num_bytes in self.components.items() if component not in HOST_COMPONENTS)

    def host_bytes(self) -> int:
        """Sum of host-resident component bytes on this rank."""
        return sum(num_bytes for component, num_bytes in self.components.items() if component in HOST_COMPONENTS)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "components": {c.value: b for c, b in sorted(self.components.items(), key=lambda kv: kv[0].value)},
            "device_bytes": self.device_bytes(),
            "host_bytes": self.host_bytes(),
            "free_bytes": self.free_bytes,
            "peak_bytes": self.peak_bytes,
        }


@dataclass
class MemoryAccountant:
    """Aggregates :class:`RankMemoryReport` across all ranks and validates balance."""

    world_size: int
    ranks: dict[int, RankMemoryReport] = field(default_factory=dict)

    def rank_report(self, rank: int) -> RankMemoryReport:
        """Return (creating if needed) the report for ``rank``."""
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} out of range for world_size {self.world_size}")
        return self.ranks.setdefault(rank, RankMemoryReport(rank=rank))

    def imbalance(self) -> float:
        """Max relative deviation of per-rank device bytes from the mean.

        Returns 0.0 when fewer than two ranks have reported or the mean is zero.
        """
        device_totals = [r.device_bytes() for r in self.ranks.values()]
        if len(device_totals) < 2:
            return 0.0
        mean = sum(device_totals) / len(device_totals)
        if mean == 0:
            return 0.0
        return max(abs(total - mean) / mean for total in device_totals)

    def validate_balance(self, approved_reason: str | None = None) -> None:
        """Raise :class:`PlacementImbalanceError` if imbalance exceeds the threshold.

        Supplying a non-empty ``approved_reason`` permits an over-threshold
        placement (the reason is recorded by the caller in the run log).
        """
        imbalance = self.imbalance()
        if imbalance > MAX_PLACEMENT_IMBALANCE and not approved_reason:
            raise PlacementImbalanceError(
                f"per-rank device placement imbalance {imbalance:.2%} exceeds "
                f"{MAX_PLACEMENT_IMBALANCE:.0%}; provide an approved_reason to override. "
                f"Per-rank device bytes: "
                f"{ {r.rank: r.device_bytes() for r in sorted(self.ranks.values(), key=lambda x: x.rank)} }"
            )

    def host_table_bytes(self) -> int:
        """Distinct host-resident bytes (the PLE table is one shared logical copy).

        Host components must be identical across ranks; this returns the single
        shared value rather than a per-rank sum, so callers never over-count the
        95.43 GiB PLE table by ``world_size``.
        """
        host_totals = {r.host_bytes() for r in self.ranks.values() if r.host_bytes() > 0}
        if not host_totals:
            return 0
        if len(host_totals) > 1:
            raise ValueError(
                f"host component bytes differ across ranks ({sorted(host_totals)}); "
                "the PLE host table must be a single shared logical copy, not per-rank copies"
            )
        return next(iter(host_totals))

    def to_dict(self) -> dict:
        return {
            "world_size": self.world_size,
            "imbalance": self.imbalance(),
            "max_imbalance_threshold": MAX_PLACEMENT_IMBALANCE,
            "host_table_bytes": self.host_table_bytes(),
            "ranks": [r.to_dict() for r in sorted(self.ranks.values(), key=lambda x: x.rank)],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def human_summary(self) -> str:
        """Render a per-rank GiB breakdown plus the shared host table line."""
        lines = [
            f"Qwen4Exp 310P memory accounting (world_size={self.world_size}, "
            f"imbalance={self.imbalance():.2%} / limit {MAX_PLACEMENT_IMBALANCE:.0%})"
        ]
        for report in sorted(self.ranks.values(), key=lambda x: x.rank):
            lines.append(f"  rank {report.rank}: device={_gib(report.device_bytes())}")
            for component, num_bytes in sorted(report.components.items(), key=lambda kv: kv[0].value):
                if component in HOST_COMPONENTS:
                    continue
                lines.append(f"      {component.value:<20} {_gib(num_bytes)}")
            if report.free_bytes is not None:
                lines.append(f"      {'free':<20} {_gib(report.free_bytes)}")
            if report.peak_bytes is not None:
                lines.append(f"      {'peak':<20} {_gib(report.peak_bytes)}")
        host_bytes = self.host_table_bytes()
        if host_bytes:
            lines.append(f"  host PLE table (shared): {_gib(host_bytes)}")
        return "\n".join(lines)


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / _BYTES_PER_GIB:.2f} GiB"
