"""Summarize four-rank GLM 310P profiler exports without adding task times.

Use ``--windows`` for explicit prefill/decode or per-step timestamp ranges. A
window's cross-rank envelope is elapsed device time, while summed task and
communication counters are attribution only. Kernel CSVs alone do not expose
the complete inter-stream dependency graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import regex as re

RANK_PATTERN = re.compile(r"(?:^|_)rank(\d+)(?:_|$)")
NANOSECONDS_PER_MICROSECOND = 1000
NANOSECONDS_PER_MILLISECOND = 1_000_000
TOP_KERNEL_SHAPES = 12


def microseconds_to_ns(value: str | int | float) -> int:
    return int(Decimal(str(value).strip()) * NANOSECONDS_PER_MICROSECOND)


def milliseconds_to_ns(value: str | int | float) -> int:
    return int(Decimal(str(value).strip()) * NANOSECONDS_PER_MILLISECOND)


def rank_from_path(path: Path) -> int:
    for part in path.parts:
        match = RANK_PATTERN.search(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot identify rank from {path}")


def kernel_category(name: str, accelerator_core: str = "") -> str:
    compact = name.lower().replace("_", "")
    if (compact == "cast" or "castaicpu" in compact) and accelerator_core == "AI_CPU":
        return "ai_cpu_cast"
    if "glmfusedgateup" in compact:
        return "moe_gate_up"
    if "glmfuseddown" in compact:
        return "moe_down"
    if "glmfusedreduce" in compact:
        return "moe_reduce"
    if "glmfusedpack" in compact or "glmfusedrouteinput" in compact:
        return "moe_prepare"
    if "w2groupedblockeddequantmatmul" in compact:
        return "grouped_w2_w4"
    if any(token in compact for token in ("hcom", "hccl", "allreduce", "allgather", "reducescatter")):
        return "collective"
    if "mhcsinkhorn" in compact:
        return "mhc_sinkhorn"
    if "qsa" in compact or "sparseattention" in compact:
        return "sparse_attention"
    if any(token in compact for token in ("kda", "gateddelta", "causalconv")):
        return "kda_or_convolution"
    if "topk" in compact:
        return "topk_all"
    if any(token in compact for token in ("bitwiseand", "rightshift", "leftshift")):
        return "integer_bit_ops"
    if any(token in compact for token in ("copy", "memcpy", "slice", "transpose", "transdata")):
        return "transfer_or_layout"
    if "matmul" in compact or compact == "mm":
        return "other_matmul"
    return "other"


def read_kernels(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    truncated_trailing_rows = 0
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            # CANN may leave a partially emitted final CSV row even after
            # reporting that its text export has finished. Accept only that
            # trailing fragment; a malformed row in the middle is an error.
            if not row.get("Start Time(us)") or not row.get("Duration(us)"):
                truncated_trailing_rows += 1
                continue
            if truncated_trailing_rows:
                raise ValueError(f"malformed kernel row before complete rows in {path}")
            start = microseconds_to_ns(row["Start Time(us)"])
            duration = microseconds_to_ns(row["Duration(us)"])
            if duration < 0:
                raise ValueError(f"negative task duration in {path}")
            events.append(
                {
                    "start_ns": start,
                    "end_ns": start + duration,
                    "name": row["Name"],
                    "category": kernel_category(row["Name"], row.get("Accelerator Core", "")),
                    "input_shapes": row.get("Input Shapes", ""),
                    "device_id": row.get("Device_id", ""),
                }
            )
    if truncated_trailing_rows > 1:
        raise ValueError(f"more than one truncated trailing kernel row in {path}")
    devices = {event["device_id"] for event in events}
    if len(devices) > 1:
        raise ValueError(f"multiple device IDs in one rank export: {path}")
    return events


def read_collectives(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    entries = data.get("step", {}).get("collective", {})
    result = []
    for key, item in entries.items():
        if key.startswith("Total"):
            # CANN emits aggregate rows without a per-call timestamp. They
            # are not collectives and would double-count the window totals.
            continue
        timing = item["Communication Time Info"]
        result.append(
            {
                "key": key.split("@", 1)[0],
                "start_ns": microseconds_to_ns(timing["Start Timestamp(us)"]),
                "elapsed_ns": milliseconds_to_ns(timing["Elapse Time(ms)"]),
                "wait_ns": milliseconds_to_ns(timing["Wait Time(ms)"]),
                "transit_ns": milliseconds_to_ns(timing["Transit Time(ms)"]),
            }
        )
    return result


def union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    left, right = intervals[0]
    total = 0
    for start, end in intervals[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def summarize_window(
    label: str,
    start_ns: int,
    end_ns: int,
    kernels_by_rank: dict[int, list[dict[str, Any]]],
    collectives_by_rank: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    if end_ns <= start_ns:
        raise ValueError(f"invalid window {label}: end must exceed start")
    ranks = {}
    matched: dict[str, dict[int, int]] = defaultdict(dict)
    observed_starts: list[int] = []
    observed_ends: list[tuple[int, int]] = []
    for rank, events in sorted(kernels_by_rank.items()):
        clipped = [
            (max(start_ns, event["start_ns"]), min(end_ns, event["end_ns"]), event)
            for event in events
            if event["start_ns"] < end_ns and event["end_ns"] > start_ns
        ]
        groups: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
        for item in clipped:
            groups[item[2]["category"]].append(item)
            observed_starts.append(item[0])
            observed_ends.append((item[1], rank))
        category_stats = {
            category: {
                "count": len(items),
                "summed_task_ms": sum(end - begin for begin, end, _ in items) / NANOSECONDS_PER_MILLISECOND,
                "task_union_ms": union_ns([(begin, end) for begin, end, _ in items]) / NANOSECONDS_PER_MILLISECOND,
                "median_task_us": statistics.median(
                    (end - begin) / NANOSECONDS_PER_MICROSECOND for begin, end, _ in items
                ),
            }
            for category, items in sorted(groups.items())
        }
        kernel_shapes: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for begin, end, event in clipped:
            kernel_shapes[(event["name"], event["input_shapes"])].append((begin, end))
        top_kernel_shapes = sorted(
            (
                {
                    "name": name,
                    "input_shapes": shapes,
                    "count": len(intervals),
                    "summed_task_ms": sum(end - begin for begin, end in intervals) / NANOSECONDS_PER_MILLISECOND,
                    "task_union_ms": union_ns(intervals) / NANOSECONDS_PER_MILLISECOND,
                }
                for (name, shapes), intervals in kernel_shapes.items()
            ),
            key=lambda item: (-item["summed_task_ms"], item["name"], item["input_shapes"]),
        )[:TOP_KERNEL_SHAPES]
        comms = [item for item in collectives_by_rank[rank] if start_ns <= item["start_ns"] < end_ns]
        for item in comms:
            if item["key"] in matched and rank in matched[item["key"]]:
                raise ValueError(f"duplicate collective key {item['key']} on rank {rank}")
            matched[item["key"]][rank] = item["start_ns"]
        ranks[str(rank)] = {
            "task_count": len(clipped),
            "summed_task_ms": sum(end - begin for begin, end, _ in clipped) / NANOSECONDS_PER_MILLISECOND,
            "task_union_ms": union_ns([(begin, end) for begin, end, _ in clipped]) / NANOSECONDS_PER_MILLISECOND,
            "categories": category_stats,
            "top_kernel_shapes": top_kernel_shapes,
            "collective_count": len(comms),
            "collective_summed_wait_ms": sum(item["wait_ns"] for item in comms) / NANOSECONDS_PER_MILLISECOND,
            "collective_summed_transit_ms": sum(item["transit_ns"] for item in comms) / NANOSECONDS_PER_MILLISECOND,
        }
    complete_spreads = [
        (max(starts.values()) - min(starts.values())) / NANOSECONDS_PER_MILLISECOND
        for starts in matched.values()
        if len(starts) == len(kernels_by_rank)
    ]
    envelope_start = min(observed_starts) if observed_starts else None
    envelope_end, tail_rank = max(observed_ends) if observed_ends else (None, None)
    return {
        "label": label,
        "window_start_us": str(Decimal(start_ns) / NANOSECONDS_PER_MICROSECOND),
        "window_end_us": str(Decimal(end_ns) / NANOSECONDS_PER_MICROSECOND),
        "cross_rank_task_envelope_ms": (
            (envelope_end - envelope_start) / NANOSECONDS_PER_MILLISECOND
            if envelope_start is not None and envelope_end is not None
            else None
        ),
        "tail_rank": tail_rank,
        "collective_arrival_spread_median_ms": statistics.median(complete_spreads) if complete_spreads else None,
        "collective_arrival_spread_max_ms": max(complete_spreads) if complete_spreads else None,
        "collectives_matched_all_ranks": len(complete_spreads),
        "collectives_missing_ranks": sum(len(starts) != len(kernels_by_rank) for starts in matched.values()),
        "ranks": ranks,
    }


def analyze(trace_root: Path, capture_token: str, windows: list[dict[str, Any]], expected_ranks: int = 4) -> dict:
    paths = [path for path in trace_root.rglob("kernel_details.csv") if capture_token in str(path.parent.parent.name)]
    if not paths:
        raise ValueError(f"no kernel_details.csv under {trace_root} for capture token {capture_token!r}")
    kernels_by_rank = {}
    collectives_by_rank = {}
    for path in sorted(paths):
        rank = rank_from_path(path)
        if rank in kernels_by_rank:
            raise ValueError(f"multiple traces for rank {rank}; narrow --capture-token")
        kernels_by_rank[rank] = read_kernels(path)
        comm_path = path.with_name("communication.json")
        collectives_by_rank[rank] = read_collectives(comm_path) if comm_path.exists() else []
    if set(kernels_by_rank) != set(range(expected_ranks)):
        raise ValueError(f"expected ranks 0..{expected_ranks - 1}, found {sorted(kernels_by_rank)}")
    if not windows:
        all_events = [event for events in kernels_by_rank.values() for event in events]
        if not all_events:
            raise ValueError("no kernel events")
        windows = [
            {
                "label": "capture",
                "start_us": str(Decimal(min(event["start_ns"] for event in all_events)) / NANOSECONDS_PER_MICROSECOND),
                "end_us": str(Decimal(max(event["end_ns"] for event in all_events)) / NANOSECONDS_PER_MICROSECOND),
            }
        ]
    return {
        "capture_token": capture_token,
        "trace_root": str(trace_root),
        "metric_warning": (
            "Summed task/collective times can overlap and are attribution only. "
            "The cross-rank task envelope and tail rank locate elapsed-time bounds, "
            "not a complete dependency-chain critical path. TopK and integer bit-op "
            "labels identify kernel names, not their calling model component. "
            "Use explicit per-step windows and host/operator annotations."
        ),
        "windows": [
            summarize_window(
                item["label"],
                microseconds_to_ns(item["start_us"]),
                microseconds_to_ns(item["end_us"]),
                kernels_by_rank,
                collectives_by_rank,
            )
            for item in windows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path, help="root containing rank profiler exports")
    parser.add_argument("--capture-token", required=True, help="substring identifying one four-rank capture")
    parser.add_argument("--windows", type=Path, help="JSON list of {label,start_us,end_us} phase/step windows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    windows = json.loads(args.windows.read_text()) if args.windows else []
    report = analyze(args.trace_root, args.capture_token, windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"capture_token": args.capture_token, "windows": len(report["windows"])}))


if __name__ == "__main__":
    main()
