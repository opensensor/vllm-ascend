#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Capture optional Ascend DMA events or summarize a saved capture."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


EVENT = "ascend_dma_sync"
FIELDS = {
    "device", "channel", "nodes", "direction", "wait_mode", "status", "bytes",
    "start_ns", "submit_ns", "wait_ns", "complete_ns", "end_ns", "sleep_ns",
    "sleeps", "queries",
}
SPIN_FIELDS = {"spin_budget_us", "spin_queries", "spin_completed", "spin_ns"}
GROUP_FIELDS = ("device", "channel", "direction", "bytes", "wait_mode", "spin_budget_us")


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1],
        "max": ordered[-1],
    }


def summarize(trace):
    groups = defaultdict(list)
    failures = Counter()
    malformed = 0
    events = 0
    for line in trace.splitlines():
        if f"{EVENT}: " not in line:
            continue
        events += 1
        payload = line.split(f"{EVENT}: ", 1)[1]
        pairs = re.findall(r"(\w+)=(-?\d+)(?=\s|$)", payload)
        row = {key: int(value) for key, value in pairs}
        if set(row) not in (FIELDS, FIELDS | SPIN_FIELDS) or len(pairs) != len(row):
            malformed += 1
            continue
        for field in SPIN_FIELDS:
            row.setdefault(field, 0)
        if any(value < 0 for key, value in row.items() if key not in {"status", "direction"}):
            malformed += 1
            continue
        if row["end_ns"] < row["start_ns"] or not row["start_ns"]:
            malformed += 1
            continue
        if row["status"] != 0:
            failures[str(row["status"])] += 1
            continue
        start, submit, wait, complete, end = (row[f"{name}_ns"] for name in
                                            ("start", "submit", "wait", "complete", "end"))
        # Completion can precede entry into the wait function.
        if not (start <= submit <= complete <= end and submit <= wait <= end):
            malformed += 1
            continue
        if row["sleep_ns"] + row["spin_ns"] > end - wait or row["spin_completed"] not in (0, 1):
            malformed += 1
            continue
        row.update(
            total_ns=end - start,
            prepare_ns=submit - start,
            submit_to_observed_complete_ns=complete - submit,
            observed_complete_to_return_ns=end - complete,
            wait_elapsed_ns=end - wait,
        )
        groups[tuple(row[key] for key in GROUP_FIELDS)].append(row)
    metrics = (
        "total_ns", "prepare_ns", "submit_to_observed_complete_ns",
        "observed_complete_to_return_ns", "wait_elapsed_ns", "sleep_ns", "sleeps", "queries",
        "spin_ns", "spin_queries", "spin_completed",
    )
    return {
        "schema": "ascend-dma-profile-v2",
        "events": events,
        "malformed_or_incomplete": malformed,
        "failure_status_counts": dict(failures),
        "groups": [
            dict(zip(GROUP_FIELDS, key), samples=len(rows),
                 metrics={metric: distribution([row[metric] for row in rows]) for metric in metrics})
            for key, rows in sorted(groups.items())
        ],
        "notes": [
            "Host software timestamps; completion includes interrupt/poll observation delay.",
            "Wait and sleep durations overlap the other phases; do not add them to total.",
            "Only successful, complete samples enter percentiles; failures are counted separately.",
            "Tracing perturbs timings. Compare throughput/latency with the event disabled too.",
        ],
    }


def capture(output, seconds, buffer_kb, event_filter, trace_root=Path("/sys/kernel/tracing")):
    if os.geteuid() != 0:
        raise PermissionError("Capture needs root access to tracefs; analyze does not.")
    if not (trace_root / "events" / "ascend_dma" / EVENT / "enable").exists():
        raise RuntimeError("The loaded driver has no ascend_dma_sync event; load the profiling build first.")
    output.mkdir(parents=True, exist_ok=False)
    instance = trace_root / "instances" / f"ascend_dma_{uuid.uuid4().hex}"
    instance.mkdir()
    event = instance / "events" / "ascend_dma" / EVENT
    try:
        (instance / "tracing_on").write_text("0\n")
        (instance / "buffer_size_kb").write_text(f"{buffer_kb}\n")
        if event_filter:
            (event / "filter").write_text(event_filter + "\n")
        (event / "enable").write_text("1\n")
        (instance / "tracing_on").write_text("1\n")
        print(f"Recording {seconds:g}s to {output}; run the workload now.", flush=True)
        time.sleep(seconds)
        (event / "enable").write_text("0\n")
        (instance / "tracing_on").write_text("0\n")
        trace = (instance / "trace").read_text()
        (output / "trace.txt").write_text(trace)
        stats = {p.parent.name: p.read_text() for p in sorted((instance / "per_cpu").glob("cpu*/stats"))}
        lost = sum(int(match.group(1)) for text in stats.values() for match in
                   re.finditer(r"^(?:overrun|commit overrun|dropped events):\s*(\d+)", text, re.MULTILINE))
        (output / "capture.json").write_text(json.dumps({
            "kernel": os.uname().release, "seconds": seconds, "buffer_kb_per_cpu": buffer_kb,
            "filter": event_filter, "lost_or_overwritten_events": lost,
            "per_cpu_stats": stats,
        }, indent=2) + "\n")
        result = summarize(trace)
        result["lost_or_overwritten_events"] = lost
        result["loss_accounting_available"] = bool(stats)
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        # Touch only our own instance. Do not change another tracer or its filters.
        try:
            (event / "enable").write_text("0\n")
            (instance / "tracing_on").write_text("0\n")
        finally:
            instance.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("capture")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--seconds", type=float, default=10)
    record.add_argument("--buffer-kb", type=int, default=256)
    record.add_argument("--filter", default="")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
            parser.error("--seconds must be in (0, 300]")
        if not 1 <= args.buffer_kb <= 4096:
            parser.error("--buffer-kb must be in [1, 4096]")
        capture(args.output, args.seconds, args.buffer_kb, args.filter)
    else:
        result = summarize(args.trace.read_text())
        result["loss_accounting_available"] = False
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
