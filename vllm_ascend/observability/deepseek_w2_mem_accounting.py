# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank memory accounting for the DeepSeek V4.1 552B W2-on-310P path.

This is a thin DeepSeek-specific layer over the Qwen4Exp per-rank accountant in
``vllm_ascend.observability.qwen38_mem_accounting`` (which it imports and reuses,
never edits). It adds the components unique to the 2-bit-expert DeepSeek
deployment on four-chip Ascend 310P:

* ``W2_EXPERT``    -- packed 2-bit expert weights, device (HBM) resident, sharded
                      across ranks and counted toward the per-chip NPU budget and
                      the placement-imbalance check.
* ``UNPACK_CACHE`` -- the INT8 active-expert unpack cache, device (HBM) resident,
                      also counted per rank.
* ``ENGRAM_HOST``  -- the ~W4 Engram host table. Like the Qwen PLE host table it
                      is a single shared logical copy in host RAM: it is excluded
                      from per-rank device totals and must NOT be multiplied by
                      ``world_size``. Divergence across ranks is rejected.

Existing Qwen device/host components (``MemComponent``) remain usable on the same
report -- e.g. non-expert FP16 weights or the shared PLE table -- so a DeepSeek
report can mix both enums. Device-resident bytes still obey the fixed 5% per-rank
placement-imbalance guard.

The module is pure Python (no torch / torch-npu). Real device statistics are
injected by the caller; the harness never queries the accelerator itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm_ascend.observability.qwen38_mem_accounting import (
    HOST_COMPONENTS as _QWEN_HOST_COMPONENTS,
)
from vllm_ascend.observability.qwen38_mem_accounting import (
    MAX_PLACEMENT_IMBALANCE,
    MemComponent,
    MemoryAccountant,
    PlacementImbalanceError,
    RankMemoryReport,
    _gib,
)

__all__ = [
    "MAX_PLACEMENT_IMBALANCE",
    "MemComponent",
    "PlacementImbalanceError",
    "DeepSeekW2MemComponent",
    "DEEPSEEK_HOST_COMPONENTS",
    "HOST_COMPONENTS",
    "DeepSeekW2RankMemoryReport",
    "DeepSeekW2MemoryAccountant",
]


class DeepSeekW2MemComponent(str, Enum):
    """DeepSeek-W2-specific memory components, layered on top of ``MemComponent``.

    ``ENGRAM_HOST`` is host-resident (a single shared logical copy, like the Qwen
    PLE table); ``W2_EXPERT`` and ``UNPACK_CACHE`` are device-resident and count
    toward the per-rank NPU budget and imbalance check.
    """

    W2_EXPERT = "w2_expert"
    UNPACK_CACHE = "unpack_cache"
    ENGRAM_HOST = "engram_host"


# DeepSeek host-resident components (shared across ranks, excluded from per-rank
# device totals and the imbalance check).
DEEPSEEK_HOST_COMPONENTS: frozenset[DeepSeekW2MemComponent] = frozenset({DeepSeekW2MemComponent.ENGRAM_HOST})

# Combined host set: the Qwen PLE table plus the DeepSeek Engram table. A single
# report may legitimately carry both host components (both are single shared
# copies), so classification unions the two registries.
HOST_COMPONENTS: frozenset = _QWEN_HOST_COMPONENTS | DEEPSEEK_HOST_COMPONENTS


class DeepSeekW2RankMemoryReport(RankMemoryReport):
    """A :class:`RankMemoryReport` that also classifies DeepSeek host components.

    The base report keys host-vs-device off the Qwen ``HOST_COMPONENTS`` set,
    which knows only the PLE table. This override consults the combined
    :data:`HOST_COMPONENTS` so that ``ENGRAM_HOST`` is treated as host-resident
    (kept out of ``device_bytes``) while ``W2_EXPERT`` / ``UNPACK_CACHE`` stay on
    the device side.
    """

    def device_bytes(self) -> int:
        """Sum of device-resident component bytes (excludes all host components)."""
        return sum(num_bytes for component, num_bytes in self.components.items() if component not in HOST_COMPONENTS)

    def host_bytes(self) -> int:
        """Sum of host-resident component bytes (PLE and/or Engram) on this rank."""
        return sum(num_bytes for component, num_bytes in self.components.items() if component in HOST_COMPONENTS)


@dataclass
class DeepSeekW2MemoryAccountant(MemoryAccountant):
    """DeepSeek-W2 accountant: reuses the Qwen aggregation/balance logic.

    Only two things differ from the base accountant: reports are
    :class:`DeepSeekW2RankMemoryReport` (so Engram is host-classified) and the
    human summary / single-copy error message are DeepSeek-worded. Imbalance,
    validation, JSON export and device-total handling are inherited unchanged.
    """

    ranks: dict[int, DeepSeekW2RankMemoryReport] = field(default_factory=dict)

    def rank_report(self, rank: int) -> DeepSeekW2RankMemoryReport:
        """Return (creating if needed) the DeepSeek report for ``rank``."""
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} out of range for world_size {self.world_size}")
        return self.ranks.setdefault(rank, DeepSeekW2RankMemoryReport(rank=rank))

    def host_table_bytes(self) -> int:
        """Distinct host-resident bytes (Engram/PLE are single shared copies).

        Host components must be identical across ranks; this returns the single
        shared value rather than a per-rank sum, so callers never over-count the
        Engram host table by ``world_size``.
        """
        host_totals = {r.host_bytes() for r in self.ranks.values() if r.host_bytes() > 0}
        if not host_totals:
            return 0
        if len(host_totals) > 1:
            raise ValueError(
                f"host component bytes differ across ranks ({sorted(host_totals)}); "
                "the Engram host table must be a single shared logical copy, not "
                "per-rank copies"
            )
        return next(iter(host_totals))

    def human_summary(self) -> str:
        """Render a per-rank GiB breakdown plus the shared Engram host line."""
        lines = [
            f"DeepSeek V4.1 W2 310P memory accounting (world_size={self.world_size}, "
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
            lines.append(f"  host Engram table (shared): {_gib(host_bytes)}")
        return "\n".join(lines)
