# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render named Ascend kernel CSVs as standalone PNG diagnostics."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tools.qwen4exp.summarize_trace import category, to_ns

COLORS = {
    "w4a8_native_int4_projection": "#3766a0",
    "w4_activation_pack": "#65a5c8",
    "moe_finalization": "#82bfd2",
    "qsa_attention": "#ca6a34",
    "other_attention_or_recurrence": "#e9aa70",
    "collective": "#9870aa",
    "cast_layout_or_copy": "#71947b",
    "other_matrix_multiply": "#d1a049",
    "other": "#9da3a8",
}


def display_category(name: str) -> str:
    compact = name.lower().replace("_", "")
    if "qwenw4a8int4matmul" in compact:
        return "w4a8_native_int4_projection"
    if "qwenw4a8pack" in compact:
        return "w4_activation_pack"
    if "moefinalizerouting" in compact:
        return "moe_finalization"
    # These two generic names carry QSA tile shapes in this 310P trace.
    if "qsa" in compact or name in {"BatchMatMul9/v2", "trans_TransData_11"}:
        return "qsa_attention"
    fallback = category(name)
    if fallback in {"named_attention_or_recurrence"}:
        return "other_attention_or_recurrence"
    if fallback in {"w4_device_grouped_projection", "w4_routed_projection", "w4_group_projection"}:
        return "w4a8_native_int4_projection"
    return fallback


def load_tasks(trace_root: Path) -> list[tuple[str, list[dict]]]:
    traces = []
    for path in sorted(trace_root.rglob("kernel_details.csv")):
        tasks = []
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                start = to_ns(row["Start Time(us)"])
                duration = to_ns(row["Duration(us)"])
                if duration < 0:
                    raise ValueError(f"negative task duration in {path}")
                tasks.append(
                    {
                        "name": row["Name"],
                        "category": display_category(row["Name"]),
                        "start": start,
                        "duration": duration,
                        "device": row["Device_id"],
                    }
                )
        if tasks:
            devices = {task["device"] for task in tasks}
            if len(devices) != 1:
                raise ValueError(f"trace contains multiple device IDs: {path}")
            traces.append((f"NPU {next(iter(devices))}", tasks))
    if not traces:
        raise ValueError(f"no populated kernel_details.csv under {trace_root}")
    return traces


def plot_mix(traces: list[tuple[str, list[dict]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, max(3.4, 0.8 * len(traces) + 1.8)))
    totals = [
        {key: sum(task["duration"] for task in tasks if task["category"] == key) / 1e6 for key in COLORS}
        for _, tasks in traces
    ]
    left = [0.0] * len(traces)
    for key, color in COLORS.items():
        widths = [row[key] for row in totals]
        if not any(widths):
            continue
        ax.barh(range(len(traces)), widths, left=left, color=color, label=key.replace("_", " "), height=0.65)
        left = [start + width for start, width in zip(left, widths, strict=True)]
    ax.set_yticks(range(len(traces)), [label for label, _ in traces])
    ax.invert_yaxis()
    ax.set_xlabel("Summed kernel task duration (ms); overlapping work is counted more than once")
    ax.set_title("Qwen cold prefill: device task mix by NPU")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_top(traces: list[tuple[str, list[dict]]], output: Path, limit: int = 18) -> None:
    totals = defaultdict(int)
    counts = defaultdict(int)
    for _, tasks in traces:
        for task in tasks:
            totals[task["name"]] += task["duration"]
            counts[task["name"]] += 1
    selected = sorted(totals, key=totals.get, reverse=True)[:limit]
    selected.reverse()
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.36 * len(selected) + 1.5)))
    ax.barh(
        range(len(selected)),
        [totals[name] / 1e6 for name in selected],
        color=[COLORS[display_category(name)] for name in selected],
    )
    labels = [f"{name[:60]}{'…' if len(name) > 60 else ''}  ({counts[name]} calls)" for name in selected]
    ax.set_yticks(range(len(selected)), labels, fontsize=8)
    ax.set_xlabel("Summed kernel task duration across captured NPUs (ms)")
    ax.set_title("Qwen cold prefill: highest-cost named kernels")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(traces: list[tuple[str, list[dict]]], output: Path) -> None:
    origin = min(task["start"] for _, tasks in traces for task in tasks)
    fig, ax = plt.subplots(figsize=(15, max(4.2, 1.1 * len(traces) + 1.5)))
    offsets = {key: (i - (len(COLORS) - 1) / 2) * 0.075 for i, key in enumerate(COLORS)}
    for rank, (_, tasks) in enumerate(traces):
        by_category = defaultdict(list)
        for task in tasks:
            left = (task["start"] - origin) / 1e6
            right = (task["start"] + task["duration"] - origin) / 1e6
            y = rank + offsets[task["category"]]
            by_category[task["category"]].append(((left, y), (right, y)))
        for key, segments in by_category.items():
            ax.add_collection(
                LineCollection(
                    segments, colors=COLORS[key], linewidths=1.2, label=key.replace("_", " ") if rank == 0 else None
                )
            )
    ax.autoscale()
    ax.set_yticks(range(len(traces)), [label for label, _ in traces])
    ax.invert_yaxis()
    ax.set_xlabel("Elapsed time from first recorded device task (ms)")
    ax.set_title("Qwen cold prefill: named task timeline (each line is one device task)")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def merge_intervals(tasks: list[dict]) -> list[tuple[int, int]]:
    intervals = sorted((task["start"], task["start"] + task["duration"]) for task in tasks if task["duration"] > 0)
    merged = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def plot_occupancy(traces: list[tuple[str, list[dict]]], output: Path, bin_ms: int = 50) -> None:
    """Show whether any device kernel runs in each time bin, across streams."""
    origin = min(task["start"] for _, tasks in traces for task in tasks)
    stop = max(task["start"] + task["duration"] for _, tasks in traces for task in tasks)
    bin_ns = bin_ms * 1_000_000
    bins = math.ceil((stop - origin) / bin_ns)
    busy = np.zeros((len(traces), bins), dtype=np.float64)
    busy_total = []
    longest_gap = []
    for rank, (_, tasks) in enumerate(traces):
        intervals = merge_intervals(tasks)
        busy_total.append(sum(right - left for left, right in intervals))
        longest_gap.append(max((right[0] - left[1] for left, right in zip(intervals, intervals[1:])), default=0))
        for left, right in intervals:
            first = (left - origin) // bin_ns
            last = (right - 1 - origin) // bin_ns
            for index in range(first, last + 1):
                bin_start = origin + index * bin_ns
                busy[rank, index] += max(0, min(right, bin_start + bin_ns) - max(left, bin_start)) / bin_ns
    fig, (ax, idle_ax) = plt.subplots(2, 1, figsize=(15, 5.2), height_ratios=(2.3, 1), sharex=False)
    span_ms = (stop - origin) / 1e6
    image = ax.imshow(
        busy,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(0, bins * bin_ms, len(traces) - 0.5, -0.5),
    )
    ax.set_xlim(0, span_ms)
    ax.set_yticks(range(len(traces)), [label for label, _ in traces])
    ax.set_title(f"Qwen cold prefill: device occupancy in {bin_ms} ms bins")
    fig.colorbar(image, ax=ax, label="Fraction of bin with any kernel running", pad=0.01)
    idle_ms = [(stop - origin - value) / 1e6 for value in busy_total]
    idle_ax.barh(range(len(traces)), idle_ms, color="#a9afb4", height=0.65)
    idle_ax.set_yticks(range(len(traces)), [label for label, _ in traces])
    idle_ax.invert_yaxis()
    idle_ax.set_xlabel("Time with no recorded device kernel (ms) in the common trace span")
    for rank, value in enumerate(idle_ms):
        idle_ax.text(
            value, rank, f"  {value:.0f} ms; longest gap {longest_gap[rank] / 1e6:.1f} ms", va="center", fontsize=8
        )
    idle_ax.set_xlim(0, max(1.0, max(idle_ms) * 1.65))
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_host_ops(operator_details: Path, output: Path, limit: int = 16) -> None:
    totals = defaultdict(float)
    counts = defaultdict(int)
    with operator_details.open(newline="") as source:
        for row in csv.DictReader(source):
            name = row["Name"]
            totals[name] += float(row["Host Self Duration(us)"] or 0) / 1000
            counts[name] += 1
    selected = sorted(totals, key=totals.get, reverse=True)[:limit]
    selected.reverse()
    fig, ax = plt.subplots(figsize=(11, max(5.0, 0.38 * len(selected) + 1.5)))
    ax.barh(range(len(selected)), [totals[name] for name in selected], color="#51778d")
    ax.set_yticks(range(len(selected)), [f"{name[:58]}  ({counts[name]} calls)" for name in selected], fontsize=8)
    ax.set_xlabel("Summed host self duration (ms); sync waits and parallel threads may overlap")
    ax.set_title("Qwen cold prefill: rank 1 host API work")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_host_copy_tail(operator_details: Path, output: Path) -> None:
    """Separate frequent cheap copy calls from rare long host API calls."""
    boundaries_us = (0, 2, 10, 100, 1_000, 10_000, math.inf)
    counts = [0] * (len(boundaries_us) - 1)
    duration_ms = [0.0] * len(counts)
    with operator_details.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["Name"] != "aclnnInplaceCopy":
                continue
            duration_us = float(row["Host Self Duration(us)"] or 0)
            for index, (lower, upper) in enumerate(zip(boundaries_us, boundaries_us[1:], strict=True)):
                if lower <= duration_us < upper:
                    counts[index] += 1
                    duration_ms[index] += duration_us / 1000
                    break
    labels = ("<2 µs", "2–10 µs", "10–100 µs", "0.1–1 ms", "1–10 ms", ">=10 ms")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, duration_ms, color="#51778d")
    for bar, count in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,} calls", ha="center", va="bottom")
    ax.set_ylabel("Summed host self duration (ms)")
    ax.set_title("Qwen cold prefill: rank 1 aclnnInplaceCopy host call tail")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--operator-details", type=Path, help="optional single-rank operator_details.csv")
    args = parser.parse_args()
    outputs = [
        args.output_dir / name
        for name in ("operator-mix.png", "top-kernels.png", "task-timeline.png", "device-occupancy.png")
    ]
    if args.operator_details:
        outputs.extend((args.output_dir / "host-ops.png", args.output_dir / "host-copy-tail.png"))
    if any(path.exists() for path in outputs):
        parser.error("one or more PNG outputs already exist; preserve prior evidence")
    traces = load_tasks(args.trace_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_mix(traces, outputs[0])
    plot_top(traces, outputs[1])
    plot_timeline(traces, outputs[2])
    plot_occupancy(traces, outputs[3])
    if args.operator_details:
        plot_host_ops(args.operator_details, outputs[4])
        plot_host_copy_tail(args.operator_details, outputs[5])
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
