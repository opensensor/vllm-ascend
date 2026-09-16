#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware / topology probe for the Qwen4Exp 1M 310P deployment (plan T0.3).

Records the facts the capacity model depends on but cannot assume from marketing
specs (PRD §5.2, open decision #1): chip inventory, *actual* free bytes per chip,
firmware/driver, HCCL device topology annotated as within-card vs cross-card,
PCIe generation/width, NUMA map, DDR channel population, and a sustained
host<->device bandwidth micro-benchmark.

The probe is structured so it is authored and unit-tested on host now and executed
unchanged on the four-chip target during the device wave (plan D1). All hardware
access goes through small injectable collector callables; the default collectors
shell out to ``npu-smi`` / read ``torch_npu`` and ``/sys``, and the unit tests
substitute fakes. The probe never imports torch-npu at module load.

Output: a versioned JSON document (schema round-trips via
:func:`ProbeReport.from_dict`) plus a human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1
_BYTES_PER_GIB = 1024**3

# A 300I Duo card exposes two 310P chips; chips whose card id matches share the
# faster within-card link. Used to classify HCCL peer links.
CHIPS_PER_300I_DUO_CARD = 2


@dataclass
class ChipInfo:
    """Per-chip inventory and memory state."""

    chip_id: int
    card_id: int
    name: str
    firmware_version: str
    driver_version: str
    total_bytes: int
    free_bytes: int
    numa_node: int | None = None

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes


@dataclass
class LinkInfo:
    """A directed HCCL peer link between two chips, classified by locality."""

    src_chip: int
    dst_chip: int
    link_class: str  # "within_card" | "cross_card"


@dataclass
class PCIeInfo:
    chip_id: int
    generation: int
    width: int


@dataclass
class BandwidthSample:
    """Sustained host<->device bandwidth for one chip, GiB/s."""

    chip_id: int
    h2d_gib_s: float
    d2h_gib_s: float


@dataclass
class ProbeReport:
    """Full probe result. Round-trips through :meth:`to_dict`/:meth:`from_dict`."""

    schema_version: int
    timestamp: float
    chips: list[ChipInfo] = field(default_factory=list)
    links: list[LinkInfo] = field(default_factory=list)
    pcie: list[PCIeInfo] = field(default_factory=list)
    ddr_channels_populated: int | None = None
    ddr_channels_total: int | None = None
    bandwidth: list[BandwidthSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, payload: dict) -> ProbeReport:
        return cls(
            schema_version=payload["schema_version"],
            timestamp=payload["timestamp"],
            chips=[ChipInfo(**c) for c in payload.get("chips", [])],
            links=[LinkInfo(**link) for link in payload.get("links", [])],
            pcie=[PCIeInfo(**p) for p in payload.get("pcie", [])],
            ddr_channels_populated=payload.get("ddr_channels_populated"),
            ddr_channels_total=payload.get("ddr_channels_total"),
            bandwidth=[BandwidthSample(**b) for b in payload.get("bandwidth", [])],
        )

    def min_free_bytes(self) -> int:
        """Smallest free-bytes across chips (the binding constraint for placement)."""
        return min((c.free_bytes for c in self.chips), default=0)

    def human_summary(self) -> str:
        lines = [f"310P hardware probe (schema v{self.schema_version})"]
        for chip in sorted(self.chips, key=lambda c: c.chip_id):
            lines.append(
                f"  chip {chip.chip_id} (card {chip.card_id}, numa {chip.numa_node}): "
                f"{chip.name} fw={chip.firmware_version} drv={chip.driver_version} "
                f"free={chip.free_bytes / _BYTES_PER_GIB:.2f} GiB / "
                f"{chip.total_bytes / _BYTES_PER_GIB:.2f} GiB"
            )
        if self.chips:
            lines.append(f"  min free per chip: {self.min_free_bytes() / _BYTES_PER_GIB:.2f} GiB")
        within = sum(1 for link in self.links if link.link_class == "within_card")
        cross = sum(1 for link in self.links if link.link_class == "cross_card")
        lines.append(f"  HCCL links: {within} within-card, {cross} cross-card")
        for pcie in sorted(self.pcie, key=lambda p: p.chip_id):
            lines.append(f"  chip {pcie.chip_id} PCIe: gen{pcie.generation} x{pcie.width}")
        if self.ddr_channels_total is not None:
            lines.append(f"  DDR channels: {self.ddr_channels_populated}/{self.ddr_channels_total} populated")
        for bw in sorted(self.bandwidth, key=lambda b: b.chip_id):
            lines.append(f"  chip {bw.chip_id} bandwidth: H2D {bw.h2d_gib_s:.1f} GiB/s, D2H {bw.d2h_gib_s:.1f} GiB/s")
        return "\n".join(lines)


def classify_links(chips: Sequence[ChipInfo]) -> list[LinkInfo]:
    """Build directed peer links, classing pairs on the same card as within-card.

    Locality is derived purely from ``card_id`` so the classification is testable
    without hardware and matches TOBS's within/cross-card collective annotation.
    """
    links: list[LinkInfo] = []
    for src in chips:
        for dst in chips:
            if src.chip_id == dst.chip_id:
                continue
            link_class = "within_card" if src.card_id == dst.card_id else "cross_card"
            links.append(LinkInfo(src_chip=src.chip_id, dst_chip=dst.chip_id, link_class=link_class))
    return links


# Collector callable types. Defaults touch hardware; tests inject fakes.
ChipCollector = Callable[[], list[ChipInfo]]
PCIeCollector = Callable[[], list[PCIeInfo]]
DDRCollector = Callable[[], tuple[int | None, int | None]]
BandwidthCollector = Callable[[Sequence[ChipInfo]], list[BandwidthSample]]


def probe(
    *,
    chip_collector: ChipCollector,
    pcie_collector: PCIeCollector | None = None,
    ddr_collector: DDRCollector | None = None,
    bandwidth_collector: BandwidthCollector | None = None,
    now: Callable[[], float] = time.time,
) -> ProbeReport:
    """Assemble a :class:`ProbeReport` from the supplied collectors.

    Only ``chip_collector`` is required; the others default to empty results so a
    partial probe (e.g. before the bandwidth micro-bench is wired on a new target)
    still produces a schema-valid document.
    """
    chips = chip_collector()
    pcie = pcie_collector() if pcie_collector is not None else []
    ddr_populated, ddr_total = ddr_collector() if ddr_collector is not None else (None, None)
    bandwidth = bandwidth_collector(chips) if bandwidth_collector is not None else []
    return ProbeReport(
        schema_version=SCHEMA_VERSION,
        timestamp=now(),
        chips=chips,
        links=classify_links(chips),
        pcie=pcie,
        ddr_channels_populated=ddr_populated,
        ddr_channels_total=ddr_total,
        bandwidth=bandwidth,
    )


def _default_chip_collector() -> list[ChipInfo]:  # pragma: no cover - hardware path
    """Collect chip inventory from ``npu-smi`` / ``torch_npu`` on the target.

    Intentionally minimal and defensive: this runs only on the device wave (D1).
    The parsing details are finalized against the pinned container per AGENTS.md;
    until then it raises so callers must supply real output.
    """
    raise NotImplementedError(
        "default chip collector must be finalized against the pinned CANN container on target (plan D1); "
        "supply a chip_collector explicitly until then"
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="310P hardware/topology probe (plan T0.3/D1)")
    parser.add_argument("--json-out", help="write the JSON report to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress the human summary")
    args = parser.parse_args(argv)

    report = probe(chip_collector=_default_chip_collector)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(report.to_json())
    if not args.quiet:
        print(report.human_summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
