# SPDX-License-Identifier: Apache-2.0
"""Summarize per-rank CANN exports within explicitly recorded model steps."""

import csv
import hashlib
import json
from collections import defaultdict
from contextlib import suppress
from decimal import Decimal
from pathlib import Path


def ns(value):
    return int(Decimal(value.strip()) * 1000)


def category(name, task_type=""):
    if task_type == "AI_CPU":
        return "ai_cpu_cast" if name == "Cast" else "ai_cpu_other"
    key = name.lower().replace("_", "")
    if "hccl" in key or "hcom" in key or "allreduce" in key:
        return "collective"
    if "glmfusedgateup" in key or "glmfuseddown" in key:
        return "native_int4_expert"
    if "glmfusedpack" in key:
        return "native_activation_pack"
    if "w2groupedblocked" in key:
        return "expert_projection"
    if "sinkhorn" in key:
        return "sinkhorn"
    if "mhcpost" in key:
        return "mhc_post"
    if "kpool" in key:
        return "kpool"
    if "topk" in key:
        return "topk"
    if "sparseattention" in key or "qsa" in key:
        return "sparse_attention"
    if "kda" in key or "delta" in key or "causalconv" in key:
        return "kda_convolution"
    if any(k in key for k in ("copy", "transdata", "cast", "memcpy")):
        return "copy_cast_layout"
    if "matmul" in key:
        return "other_matmul"
    return "other"


def union(intervals):
    total = 0
    previous = None
    for begin, end in sorted(intervals):
        if previous is None:
            previous = [begin, end]
        elif begin <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            total += previous[1] - previous[0]
            previous = [begin, end]
    return total + (previous[1] - previous[0] if previous else 0)


def summarize(events, begin, end):
    groups = defaultdict(list)
    names = defaultdict(list)
    selected = []
    for event in events:
        low, high = max(begin, event[0]), min(end, event[1])
        if high <= low:
            continue
        pair = (low, high)
        selected.append(pair)
        groups[event[3]].append(pair)
        names[event[2]].append(pair)
    total_union = union(selected)
    return {
        "span_ms": (end - begin) / 1e6,
        "task_union_ms": total_union / 1e6,
        "no_reported_task_ms": ((end - begin) - total_union) / 1e6,
        "categories": {
            k: {"count": len(v), "summed_ms": sum(b - a for a, b in v) / 1e6, "union_ms": union(v) / 1e6}
            for k, v in groups.items()
        },
        "top_tasks": sorted(
            [{"name": k, "count": len(v), "summed_ms": sum(b - a for a, b in v) / 1e6} for k, v in names.items()],
            key=lambda v: -v["summed_ms"],
        )[:30],
    }


def main():
    root = Path(__file__).resolve().parent
    report = {
        "scope": "All-rank CANN task/collective attribution with explicit host model-step boundaries.",
        "limitations": [
            "Profiling perturbs latency; these are not unprofiled throughput benchmarks.",
            "Task/collective sums may overlap. No reported task is not necessarily idle hardware.",
            "CANN step markers bound iterations and are cross-checked with host stamps; no complete dependency DAG.",
            "Prefill and decode separated using scheduler token counts; decode includes MTP draft/verification.",
        ],
        "launch_blocking": False,
        "workloads": {},
    }
    for label in ("c1", "c4"):
        ranks = []
        for rank in range(4):
            directory = root / label / f"rank{rank}"
            files = list(directory.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"))
            assert len(files) == 1, files
            events = []
            metric_rows = []
            with files[0].open(newline="") as source:
                for row in csv.DictReader(source):
                    begin = ns(row["Task Start Time(us)"])
                    end = begin + ns(row["Task Duration(us)"])
                    name = row["Op Name"]
                    metric_rows.append((begin, end, row))
                    events.append((begin, end, name, category(name, row["Task Type"]), row["Stream ID"]))
            steps = json.loads((directory / "host-steps.json").read_text())
            timing_files = list(directory.glob("PROF_*/mindstudio_profiler_output/step_trace*.csv"))
            assert len(timing_files) == 1, timing_files
            with timing_files[0].open(newline="") as timing_source:
                timings = list(csv.DictReader(timing_source))
            measured_steps = [s for s in steps if s["tokens"] > 0 and "end_wall_ns" in s]
            assert len(timings) == len(measured_steps), (len(timings), len(measured_steps))
            device_times = {}
            for step, timing in zip(measured_steps, timings):
                finish = ns(timing["Iteration End(us)"])
                begin = finish - ns(timing["Iteration Time(us)"])
                # Wall stamps are a cross-check, not the device phase boundary.
                assert abs(begin - step["start_wall_ns"]) < 100_000_000
                assert -1_000_000 < finish - step["end_wall_ns"] < 1_000_000_000
                step["event_end_minus_host_end_ms"] = (finish - step["end_wall_ns"]) / 1e6
                device_times[step["step"]] = (begin, finish)
            result = {
                "rank": rank,
                "source": str(files[0]),
                "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
                "exported_tasks": len(events),
                "steps": [],
            }
            for step in steps:
                if step["tokens"] <= 0 or "end_wall_ns" not in step:
                    continue
                phase = "prefill" if step["tokens"] > 8 else "decode"
                result["steps"].append(
                    {
                        **step,
                        "phase": phase,
                        "device_start_ns": device_times[step["step"]][0],
                        "device_end_ns": device_times[step["step"]][1],
                        **summarize(events, *device_times[step["step"]]),
                    }
                )
            decode = [s for s in result["steps"] if s["phase"] == "decode"]
            assert len(decode) >= 8, (label, rank, len(decode))
            result["decode"] = summarize(events, decode[0]["device_start_ns"], decode[-1]["device_end_ns"])
            metric_groups = defaultdict(list)
            low, high = decode[0]["device_start_ns"], decode[-1]["device_end_ns"]
            for begin, end, row in metric_rows:
                overlap = max(0, min(end, high) - max(begin, low))
                if overlap:
                    metric_groups[category(row["Op Name"], row["Task Type"])].append((overlap, row))
            result["decode"]["pipeline_metrics"] = {}
            for family, rows in metric_groups.items():
                measured = {}
                for key in (
                    "vec_ratio",
                    "mac_ratio",
                    "scalar_ratio",
                    "mte1_ratio",
                    "mte2_ratio",
                    "mte3_ratio",
                    "cube_utilization(%)",
                    "memory_bound",
                ):
                    values = []
                    for duration, row in rows:
                        with suppress(KeyError, ValueError):
                            values.append((duration, float(row[key])))
                    if values:
                        measured[key] = sum(d * v for d, v in values) / sum(d for d, _ in values)
                result["decode"]["pipeline_metrics"][family] = measured
            result["decode"]["model_steps"] = len(decode)
            result["decode"]["requests_per_step"] = [s["requests"] for s in decode]
            active_requests = 1 if label == "c1" else 4
            steady = [s for s in decode if s["requests"] == active_requests]
            if not steady:
                raise ValueError("capture has no full-concurrency decode steps")
            result["steady_decode"] = summarize(events, steady[0]["device_start_ns"], steady[-1]["device_end_ns"])
            result["steady_decode"]["model_steps"] = len(steady)
            result["steady_decode"]["requests_per_step"] = [s["requests"] for s in steady]

            ownership = json.loads((root / f"native-disk-load-report-rank{rank}.json").read_text())
            banks = [bank for model in ownership["models"] for bank in model["banks"]]
            bit_costs = defaultdict(list)
            for step in steady:
                for stage in ("glm_fused_gate_up_v1", "glm_fused_down_v1"):
                    kernels = sorted(
                        (b, e, row)
                        for b, e, row in metric_rows
                        if row["Op Name"] == stage and step["device_start_ns"] <= b < step["device_end_ns"]
                    )
                    if len(kernels) != len(banks):
                        raise ValueError("native stage count differs from ordered model ownership")
                    for bank, (b, e, row) in zip(banks, kernels):
                        bits = bank["bits"] if stage == "glm_fused_gate_up_v1" else bank["down_bits"]
                        bit_costs[f"W{bits}:{stage}"].append(((e - b) / 1e6, row))
            precision = {}
            for key, values in bit_costs.items():
                total = sum(duration for duration, _ in values)
                ratios = {}
                for metric in ("vec_ratio", "mac_ratio", "scalar_ratio", "mte2_ratio"):
                    valid = []
                    for duration, row in values:
                        with suppress(KeyError, ValueError):
                            valid.append((duration, float(row[metric])))
                    if valid:
                        ratios[metric] = sum(d * v for d, v in valid) / sum(d for d, _ in valid)
                precision[key] = {
                    "count": len(values),
                    "summed_ms": total,
                    "mean_ms": total / len(values),
                    "pipeline_ratios": ratios,
                }
            result["steady_decode"]["layer_associated_precision"] = precision
            result["precision_limit"] = (
                "Precision is associated with existing layer ownership, not a controlled bit-width comparison. "
                "Layers have different activations and route occupancy. Every steady step matched all native banks."
            )
            raw_files = list(directory.glob("PROF_*/mindstudio_profiler_output/task_time*.csv"))
            if len(raw_files) != 1:
                raise ValueError("capture must have one raw task export")
            raw_events = []
            with raw_files[0].open(newline="") as raw_source:
                for row in csv.DictReader(raw_source):
                    begin = ns(row["task_start(us)"])
                    finish = ns(row["task_stop(us)"])
                    raw_events.append((begin, finish, row["kernel_name"], row["kernel_type"], row["stream_id"]))
            result["raw_decode"] = summarize(raw_events, decode[0]["device_start_ns"], decode[-1]["device_end_ns"])
            result["raw_steady_decode"] = summarize(
                raw_events, steady[0]["device_start_ns"], steady[-1]["device_end_ns"]
            )
            result["raw_source"] = str(raw_files[0])
            result["raw_sha256"] = hashlib.sha256(raw_files[0].read_bytes()).hexdigest()
            ranks.append(result)
        report["workloads"][label] = ranks
    (root / "attribution.json").write_text(json.dumps(report, indent=2) + "\n")
    for label, ranks in report["workloads"].items():
        for r in ranks:
            print(
                label,
                r["rank"],
                {k: round(v, 2) for k, v in r["decode"].items() if isinstance(v, (int, float))},
                [(v["name"], round(v["summed_ms"], 2)) for v in r["decode"]["top_tasks"][:5]],
            )


if __name__ == "__main__":
    main()
