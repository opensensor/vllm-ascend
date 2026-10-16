"""Summarize a Qwen3-VL device trace using tensor shapes to isolate text decode."""

import argparse
import collections
import csv
import json
import math
from pathlib import Path

QKV_MATMUL_TYPES = {"MatMulV2", "QuantBatchMatmulV3", "QuantBatchMatmulV4", "QuantBatchMatmulV5"}


def is_single_row_qkv(row: dict, width: int) -> bool:
    if row["Type"] not in QKV_MATMUL_TYPES:
        return False
    output = row["Output Shapes"].strip('"')
    if output == f"1,{width}":
        return True
    # NZ pads M to 16, so its output alone cannot distinguish M=1 from M=16.
    # The final per-token activation scale input proves the logical row count.
    inputs = row.get("Input Shapes", "").strip('"').split(";")
    return (
        row["Type"].startswith("QuantBatchMatmul")
        and width % 16 == 0
        and output == f"{width // 16},1,16,16"
        and len(inputs) == 4
        and inputs[-1] == "1"
    )


def interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def summarize(rows: list[dict], qkv_width: int = 6144, layers: int = 36) -> dict:
    if qkv_width <= 0 or layers <= 0:
        raise ValueError("QKV width and layer count must be positive")
    if len({row["Device_id"] for row in rows}) != 1:
        raise ValueError("Select a kernel CSV containing exactly one device")
    markers = [row for row in rows if is_single_row_qkv(row, qkv_width)]
    if not markers or len(markers) % layers:
        raise ValueError("Trace has no complete M=1 text QKV layer groups")
    origin = min(float(row["Start Time(us)"]) for row in markers)
    decode = [row for row in rows if float(row["Start Time(us)"]) >= origin]
    intervals = []
    statistics = collections.defaultdict(lambda: [0, 0.0])
    for row in decode:
        start, duration = float(row["Start Time(us)"]), float(row["Duration(us)"])
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid device task timestamp or duration")
        intervals.append((start, start + duration))
        statistics[row["Type"]][0] += 1
        statistics[row["Type"]][1] += duration
    span = max(end for _, end in intervals) - origin
    total = sum(value[1] for value in statistics.values())
    if span <= 0 or total <= 0:
        raise ValueError("Device task durations must be positive")
    active = interval_union(intervals)
    return {
        "phase_basis": (
            f"First M=1 text QKV projection (logical output 1x{qkv_width}, ND or validated NZ) "
            "through final recorded device task; "
            "vision and prefill are excluded using shapes, independently of profiler delay."
        ),
        "captured_decode_forwards": len(markers) // layers,
        "recorded_device_span_ms": span / 1000,
        "recorded_device_active_union_ms": active / 1000,
        "recorded_device_gaps_ms": (span - active) / 1000,
        "recorded_device_active_fraction": active / span,
        "kernels": len(decode),
        "kernel_sum_ms": total / 1000,
        "top_kernel_types": [
            {"type": name, "count": count, "sum_ms": duration / 1000, "kernel_time_fraction": duration / total}
            for name, (count, duration) in sorted(statistics.items(), key=lambda item: -item[1][1])[:15]
        ],
        "profiled_not_throughput_baseline": True,
        "limitations": (
            "Fractions describe recorded device tasks, not CPU latency attribution. Profiler overhead is present. "
            "The first decode normalization before the QKV marker is excluded; a partial tail may remain. "
            "This shape-based classifier is specific to the configured dense Qwen3-VL geometry. "
            "Native graph traces may omit composite communication metadata; missing types do not prove zero work."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qkv-width", type=int, default=6144)
    parser.add_argument("--layers", type=int, default=36)
    args = parser.parse_args()
    with args.kernel_csv.open() as file:
        report = summarize(list(csv.DictReader(file)), args.qkv_width, args.layers)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
