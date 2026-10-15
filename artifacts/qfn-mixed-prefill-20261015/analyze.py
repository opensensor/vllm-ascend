"""Summarize host spans, streamed progress and exported native task intervals."""

import csv
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from tools.qwen4exp.summarize_trace import union_ns

ROOT = Path(__file__).resolve().parent


def ns(value):
    return int(Decimal(str(value).strip()) * 1000)


def ranked(groups):
    return [
        dict(name=name, count=len(values), sum_ms=sum(values) / 1e6, max_ms=max(values) / 1e6)
        for name, values in sorted(groups.items(), key=lambda pair: sum(pair[1]), reverse=True)
    ]


def summarize_run(path):
    receipt = json.loads((path / "receipt.json").read_text())
    result = dict(
        unpaused=receipt["unpaused"],
        profiler_status=receipt["profiler_status"],
        scheduler_start=receipt["start"]["applied"],
        scheduler_stop=receipt["stop"]["applied"],
    )
    for name in ("decode", "prefill"):
        if name not in receipt:
            continue
        row = receipt[name]
        events = [e["time_ns"] for e in row["events"]]
        result[name] = dict(
            prompt_tokens=row["usage"]["prompt_tokens"],
            cached_tokens=row["usage"]["prompt_tokens_details"]["cached_tokens"],
            completion_tokens=row["usage"]["completion_tokens"],
            ttft_s=row["ttft_s"],
            max_content_gap_s=row["max_content_gap_s"],
            median_content_gap_s=row["median_content_gap_s"],
            total_s=(row["end_ns"] - row["start_ns"]) / 1e9,
        )
        if name == "decode" and "prefill" in receipt:
            left = receipt["prefill"]["start_ns"]
            right = left + receipt["prefill"]["ttft_s"] * 1e9
            gaps = [(b - a) / 1e9 for a, b in zip(events, events[1:]) if b >= left and a <= right]
            result[name]["during_prefill_median_content_gap_s"] = statistics.median(gaps)
            result[name]["during_prefill_content_updates"] = sum(left <= t <= right for t in events)
    result["ranks"] = []
    for source in sorted(path.glob("rank*-spans.json")):
        spans = json.loads(source.read_text())
        groups = defaultdict(list)
        for row in spans["records"]:
            groups[row["label"]].append(row["duration_ns"])
        result["ranks"].append(
            dict(
                rank=spans["rank"],
                dropped=spans["dropped"],
                spans=ranked(groups),
                mixed_batches=[
                    r
                    for r in spans["records"]
                    if r["label"] == "execute_model" and len(r["tokens"]) > 1 and max(r["tokens"].values()) > 12
                ],
            )
        )
    return result


def summarize_native(path):
    kernels = defaultdict(list)
    for row in csv.DictReader((path / "kernel_details.csv").open()):
        kernels[row["Type"]].append(ns(row["Duration(us)"]))
    tasks = list(csv.DictReader((path / "task_time.csv").open()))
    intervals = [(ns(r["task_start(us)"]), ns(r["task_stop(us)"])) for r in tasks]
    work = [
        (ns(r["task_start(us)"]), ns(r["task_stop(us)"]))
        for r in tasks
        if r["kernel_type"]
        in ("AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "KERNEL_AICPU", "MEMCPY_ASYNC", "REDUCE_ASYNC_V2")
    ]
    streams = defaultdict(list)
    kinds = defaultdict(list)
    for row, interval in zip(tasks, intervals):
        streams[row["stream_id"]].append(interval)
        kinds[row["kernel_type"]].append(interval[1] - interval[0])
    events = json.loads((path / "trace_view.json").read_text())
    if isinstance(events, dict):
        events = events["traceEvents"]
    apis = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e["name"].startswith("AscendCL@"):
            apis[e["name"]].append(ns(e["dur"]))
    left, right = min(a for a, b in intervals), max(b for a, b in intervals)
    return dict(
        path=str(path.relative_to(ROOT)),
        span_ms=(right - left) / 1e6,
        task_union_ms=union_ns(intervals) / 1e6,
        work_union_ms=union_ns(work) / 1e6,
        task_kinds=ranked(kinds),
        kernels=ranked(kernels),
        host_apis=ranked(apis),
        streams={key: dict(tasks=len(value), union_ms=union_ns(value) / 1e6) for key, value in streams.items()},
    )


def main():
    runs = {
        p.name: summarize_run(p)
        for p in sorted(ROOT.iterdir())
        if (p / "receipt.json").is_file() and "after" in json.loads((p / "receipt.json").read_text())
    }
    traces = [summarize_native(p.parent) for p in ROOT.glob("mixed-native-v2/native/**/kernel_details.csv")]
    output = dict(
        scope=(
            "Synthetic real-weight requests on the retained live server. SSE content events are not individual tokens. "
            "Host nested spans and rank totals overlap; kernel sums are not wall time or predicted savings. "
            "Native run has profiling overhead. Unrecorded-task gaps do not prove device idleness."
        ),
        runs=runs,
        native=traces,
    )
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    for name, row in runs.items():
        print(name, json.dumps({key: row[key] for key in ("decode", "prefill") if key in row}))
    for row in traces:
        print(row["path"], row["span_ms"], row["work_union_ms"], row["kernels"][:8], row["host_apis"][:5])


if __name__ == "__main__":
    main()
