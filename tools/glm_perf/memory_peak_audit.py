# SPDX-License-Identifier: Apache-2.0
"""Audit archived allocator snapshots without connecting to a serving engine.

Removing a large allocation does not remove peaks during other operators.
The counterfactual below subtracts selected, complete allocation lifetimes
from recorded snapshots only. It is neither a capacity proof nor a prediction
of changed allocator scheduling, reserved memory, or graph capture storage.
"""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class Allocation:
    name: str
    device: str
    size_mib: Decimal
    start: Decimal | None
    end: Decimal | None
    allocated_at_start: Decimal | None
    allocated_at_end: Decimal | None
    reserved_at_start: Decimal | None

    def live_at(self, timestamp):
        return self.start is not None and self.start <= timestamp and (self.end is None or timestamp < self.end)


def number(row, key):
    value = row.get(key, "").strip()
    if not value:
        return None
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError(f"invalid nonnegative finite memory field: {key}")
    return result


def read_allocations(stream):
    """Preserve microsecond ordering at epoch-sized timestamps using Decimal."""
    result = []
    for row in csv.DictReader(stream):
        size = number(row, "Size(KB)")
        if size is None:
            raise ValueError("allocation size is required")
        start, end = number(row, "Allocation Time(us)"), number(row, "Release Time(us)")
        if start is not None and end is not None and end < start:
            raise ValueError("allocation release precedes allocation")
        result.append(
            Allocation(
                row["Name"].strip(),
                row["Device Type"].strip(),
                size / 1024,
                start,
                end,
                number(row, "Allocation Total Allocated(MB)"),
                number(row, "Release Total Allocated(MB)"),
                number(row, "Allocation Total Reserved(MB)"),
            )
        )
    return result


def describe(allocation):
    return {
        "name": allocation.name,
        "size_mib": float(allocation.size_mib),
        "allocation_time_us": str(allocation.start) if allocation.start is not None else None,
        "release_time_us": str(allocation.end) if allocation.end is not None else None,
    }


def audit_device(allocations, *, selected_name, minimum_mib):
    snapshots = [
        (time, allocated, allocation, phase)
        for allocation in allocations
        for time, allocated, phase in (
            (allocation.start, allocation.allocated_at_start, "allocation"),
            (allocation.end, allocation.allocated_at_end, "release"),
        )
        if time is not None and allocated is not None
    ]
    if not snapshots:
        raise ValueError("trace has no timed allocator snapshots")
    selected = [a for a in allocations if a.name == selected_name and a.size_mib >= minimum_mib]
    complete = [a for a in selected if a.start is not None and a.end is not None]
    peak_time, peak, owner, phase = max(snapshots, key=lambda entry: entry[1])
    live = sorted((a for a in allocations if a.live_at(peak_time)), key=lambda a: a.size_mib, reverse=True)
    outside = [entry for entry in snapshots if not any(a.live_at(entry[0]) for a in complete)]
    outside_peak = max(outside, key=lambda entry: entry[1]) if outside else None
    ideal = max(
        allocated - sum((a.size_mib for a in complete if a.live_at(time)), Decimal(0))
        for time, allocated, _, _ in snapshots
    )
    return {
        "records": len(allocations),
        "incomplete_lifetimes": sum(a.start is None or a.end is None for a in allocations),
        "peak_allocated_mib": float(peak),
        "peak_time_us": str(peak_time),
        "peak_snapshot_phase": phase,
        "peak_snapshot_owner": describe(owner),
        "peak_owner_reserved_at_allocation_mib": (
            float(owner.reserved_at_start) if owner.reserved_at_start is not None else None
        ),
        "largest_recorded_allocations_live_at_peak": [describe(a) for a in live[:20]],
        "selected_operator": selected_name,
        "selected_minimum_mib": float(minimum_mib),
        "selected_records": len(selected),
        "selected_complete_lifetimes": len(complete),
        "peak_outside_complete_selected_lifetimes": (
            {"allocated_mib": float(outside_peak[1]), "snapshot_owner": describe(outside_peak[2])}
            if outside_peak
            else None
        ),
        "ideal_snapshot_peak_without_complete_selected_buffers_mib": float(ideal),
        "ideal_snapshot_peak_reduction_mib": float(peak - ideal),
        "limitations": [
            "Unchanged snapshot timing; excludes replacement buffers and any changed scheduling.",
            "Incomplete lifetimes are never subtracted; allocations preceding the trace may be missing.",
            "Allocated memory differs from reserved memory; this is not a graph or larger-chunk capacity proof.",
            "Operator names identify allocations, not their model call sites; source attribution is separate.",
        ],
    }


def audit_archive(path, *, selected_name="aten::bmm", minimum_mib=Decimal(400)):
    if not minimum_mib.is_finite() or minimum_mib <= 0:
        raise ValueError("selected minimum must be finite and positive")
    tables = []
    # Read members in memory; never extract arbitrary archived paths.
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith("/operator_memory.csv"):
                continue
            by_device = defaultdict(list)
            with io.TextIOWrapper(archive.extractfile(member), encoding="utf-8-sig") as stream:
                for allocation in read_allocations(stream):
                    by_device[allocation.device].append(allocation)
            for device, allocations in sorted(by_device.items()):
                tables.append(
                    {
                        "member": member.name,
                        "device": device,
                        **audit_device(allocations, selected_name=selected_name, minimum_mib=minimum_mib),
                    }
                )
    if not tables:
        raise ValueError("archive contains no operator memory tables")
    return {"archive": str(path), "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tables": tables}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-name", default="aten::bmm")
    parser.add_argument("--minimum-mib", type=Decimal, default=Decimal(400))
    args = parser.parse_args()
    result = audit_archive(args.archive, selected_name=args.selected_name, minimum_mib=args.minimum_mib)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
