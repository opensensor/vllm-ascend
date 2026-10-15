"""Audit graph submission boundaries and device interval unions per rank."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from tools.qwen4exp.summarize_trace import union_ns


def ns(value):
    return int(Decimal(str(value).strip()) * 1000)


def clip(intervals, left, right):
    return [(max(a, left), min(b, right)) for a, b in intervals if a < right and b > left]


def summarize(path, target_segments):
    events = json.loads((path / "trace_view.json").read_text())
    events = [e for e in events if e.get("ph") == "X"]
    launches = sorted(
        (ns(e["ts"]), ns(e["ts"]) + ns(e["dur"])) for e in events if e["name"] == "AscendCL@aclmdlRIExecuteAsync"
    )
    per_iteration = target_segments + 11
    if len(launches) != 8 * per_iteration:
        raise ValueError(f"unexpected graph count in {path}: {len(launches)}")
    # Exclude first recorder-start iteration and final incomplete boundary.
    left, right = launches[per_iteration][0], launches[7 * per_iteration][0]
    tasks = list(csv.DictReader((path / "task_time.csv").open()))
    all_tasks = [(ns(r["task_start(us)"]), ns(r["task_stop(us)"])) for r in tasks]
    work = [
        (ns(r["task_start(us)"]), ns(r["task_stop(us)"]))
        for r in tasks
        if r["kernel_type"] in ("AI_CORE", "AI_CPU", "MEMCPY_ASYNC")
    ]
    all_tasks, work = clip(all_tasks, left, right), clip(work, left, right)
    kernels, shapes, apis = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in csv.DictReader((path / "kernel_details.csv").open()):
        start, duration = ns(row["Start Time(us)"]), ns(row["Duration(us)"])
        if start >= left and start + duration <= right:
            kernels[row["Type"]].append(duration)
            if row["Type"] in ("MatMulV2", "TransData", "QuantBatchMatmulV3"):
                shapes[(row["Type"], row["Input Shapes"])].append(duration)
    for event in events:
        if event["name"].startswith("AscendCL@"):
            interval = clip([(ns(event["ts"]), ns(event["ts"]) + ns(event["dur"]))], left, right)
            if interval:
                apis[event["name"]].append(interval[0][1] - interval[0][0])

    def boundary_name(slot):
        if slot < target_segments - 1:
            return "target_ple" if slot == 0 else "target_qsa_selection"
        return (
            "target_to_acceptance", "acceptance_to_draft",
            "draft_position_copy", "draft_qsa_selection", "draft_expert_dispatch",
            "draft_to_draft", "draft_position_copy", "draft_qsa_selection",
            "draft_expert_dispatch", "draft_to_next_inputs",
            "next_inputs_to_history", "history_to_next_iteration",
        )[slot - (target_segments - 1)]

    boundaries = defaultdict(list)
    for index in range(per_iteration, 7 * per_iteration):
        a, b = launches[index][1], launches[index + 1][0]
        name = boundary_name(index % per_iteration)
        boundaries[name].append(
            dict(
                span_ms=(b - a) / 1e6,
                no_task_ms=((b - a) - union_ns(clip(all_tasks, a, b))) / 1e6,
                work_ms=union_ns(clip(work, a, b)) / 1e6,
            )
        )

    def ranked(values):
        return [
            dict(name=name, count=len(durations), sum_ms=sum(durations) / 1e6, ms_per_iteration=sum(durations) / 6e6)
            for name, durations in sorted(values.items(), key=lambda pair: sum(pair[1]), reverse=True)
        ]

    iteration_rows = []
    for n in range(1, 7):
        a, b = launches[n * per_iteration][0], launches[(n + 1) * per_iteration][0]
        iteration_rows.append(
            dict(
                index=n,
                span_ms=(b - a) / 1e6,
                no_task_ms=((b - a) - union_ns(clip(all_tasks, a, b))) / 1e6,
                work_ms=union_ns(clip(work, a, b)) / 1e6,
            )
        )
    return dict(
        path=str(path),
        recorded_graph_launches=len(launches),
        iterations=6,
        span_ms=(right - left) / 1e6,
        iteration_ms=(right - left) / 6e6,
        median_iteration_ms=statistics.median(row["span_ms"] for row in iteration_rows),
        iteration_rows=iteration_rows,
        all_task_union_ms=union_ns(all_tasks) / 1e6,
        work_union_ms=union_ns(work) / 1e6,
        no_task_ms=((right - left) - union_ns(all_tasks)) / 1e6,
        graph_submit_api_ms=union_ns(clip(launches, left, right)) / 1e6,
        iteration_intervals_ms=[
            (launches[(n + 1) * per_iteration][0] - launches[n * per_iteration][0]) / 1e6 for n in range(1, 7)
        ],
        boundaries={
            name: dict(count=len(rows), **{key: sum(row[key] for row in rows) / 6 for key in rows[0]})
            for name, rows in boundaries.items()
        },
        kernel_types=ranked(kernels),
        matrix_layout_shapes=ranked(shapes),
        host_apis=ranked(apis),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(scope="Six complete profiled iterations on ranks 0 and 3. Unrecorded-task gaps are not proof of device idleness; kernel sums are not latency savings.", cases={})
    for label in ("on-v1", "off-v1"):
        paths = sorted((args.root / label / "c1-prefix").rglob("kernel_details.csv"))
        if len(paths) != 2:
            raise ValueError("expected exactly two parsed rank traces")
        rows = [summarize(path.parent, 2) for path in paths]
        report["cases"][label] = dict(ranks=rows, median_iteration_ms=statistics.median(r["iteration_ms"] for r in rows))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for label, case in report["cases"].items():
        print(json.dumps(dict(label=label, mean_iteration_ms=case["median_iteration_ms"],
                              boundaries=case["ranks"][0]["boundaries"])), flush=True)


if __name__ == "__main__":
    main()
