"""Conservative decode-tail accounting for the bounded text-only scene traces."""

import argparse
import collections
import csv
import json
import math
from decimal import Decimal
from pathlib import Path

from tools.rtsp_scene.profile_summary import interval_union

TEXT_LAYERS = 36
PROJECTIONS_PER_LAYER = 4


def summarize_text_decode(rows, layers=TEXT_LAYERS):
    if not rows or layers < 1 or len({row["Device_id"] for row in rows}) != 1:
        raise ValueError("Select nonempty kernel rows from one device and positive layer count")

    def start(row):
        return Decimal(row["Start Time(us)"].strip())

    native = [row for row in rows if row["Type"].startswith("QuantBatchMatmul")]
    if not native:
        raise ValueError("No native text prefill boundary in this text-only trace")
    boundary = max(start(row) + Decimal(row["Duration(us)"].strip()) for row in native)
    markers = [row for row in rows if row["Type"] == "rtsp_dense_i8" and start(row) >= boundary]
    if not markers:
        raise ValueError("No custom decode matrices after the final prefill boundary")
    origin = min(start(row) for row in markers)
    tail = [row for row in rows if start(row) >= origin]
    counts = collections.Counter(row["Type"] for row in tail)
    matrices, activations = counts["rtsp_dense_i8"], counts["rtsp_nz_swiglu"]
    if matrices % (layers * PROJECTIONS_PER_LAYER) or activations != matrices // PROJECTIONS_PER_LAYER:
        raise ValueError("Decode tail lacks complete projection/activation layer groups")
    durations = collections.Counter()
    intervals = []
    for row in tail:
        offset, duration = float(start(row) - origin), float(row["Duration(us)"])
        if not math.isfinite(offset) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid device timestamp or duration")
        durations[row["Type"]] += duration
        intervals.append((offset, offset + duration))
    total, span = sum(durations.values()), max(end for _, end in intervals)
    if total <= 0 or span <= 0:
        raise ValueError("Decode task durations must be positive")
    active = interval_union(intervals)
    return {
        "device": tail[0]["Device_id"],
        "decode_forwards": matrices // (layers * PROJECTIONS_PER_LAYER),
        "custom_matrices": matrices,
        "custom_activations": activations,
        "device_span_ms": span / 1000,
        "active_union_ms": active / 1000,
        "device_gaps_ms": (span - active) / 1000,
        "task_sum_ms": total / 1000,
        "types": [
            {"type": name, "calls": counts[name], "sum_ms": value / 1000, "task_fraction": value / total}
            for name, value in durations.most_common()
        ],
        "head_output_shapes": dict(
            collections.Counter(row.get("Output Shapes") for row in tail if row["Type"] == "MatMulV2")
        ),
        "phase_basis": "First custom matrix after final native quantized prefill task; excludes preceding norm",
        "limitations": "Requires text-only input and selected four-projection/one-activation decoder. "
        "Interleaved decode before final prefill is excluded. Native graphs can omit communication metadata. "
        "Task fractions are not wall-latency or CPU fractions; profiler overhead is present.",
        "profiled_not_throughput_baseline": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    cohorts = json.loads((args.profiles / "summary.json").read_text())
    result = []
    for cohort in cohorts:
        if not cohort["scope"].startswith("Text-only"):
            raise ValueError("This classifier requires the text-only profile contract")
        for path in (args.profiles / f"c{cohort['concurrency']}").rglob("kernel_details.csv"):
            row = summarize_text_decode(list(csv.DictReader(path.open())))
            row.update(concurrency=cohort["concurrency"], path=str(path), context_tokens=cohort["context_tokens"])
            result.append(row)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k not in ("types", "path")} for row in result], indent=2))


if __name__ == "__main__":
    main()
