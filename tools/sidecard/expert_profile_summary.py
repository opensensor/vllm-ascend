# SPDX-License-Identifier: Apache-2.0
"""Join exact private expert invocations and summarize disjoint host spans."""

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def summarize(run, frames, workers):
    """Check disjoint host accounting and join each exact two-chip invocation."""
    if not run["passed"] or not run["full_vocabulary_reference_bitwise_equal"]:
        raise ValueError("profile requires completed reference parity")
    by_invocation = {}
    for worker in workers:
        key = worker["rank"], worker["sequence"], worker["layer"], worker["rows"]
        if key in by_invocation:
            raise ValueError("duplicate native profile invocation")
        by_invocation[key] = worker
    report = dict(
        passed=True,
        production_speed_validated=False,
        timing_scope="device stream spans include launch/idle gaps and overlap host waits",
        host_partitions_disjoint=True,
        workloads=[],
    )
    for workload in run["workloads"]:
        selected = [
            frame for frame in frames if workload["first_sequence"] <= frame["sequence"] < workload["end_sequence"]
        ]
        result = dict(name=workload["name"], phases={})
        for phase in ("prefill", "decode"):
            group = [
                frame for frame in selected if (frame["position"] < workload["prompt_tokens"]) == (phase == "prefill")
            ]
            if not group:
                raise ValueError("missing observed prefill or decode")
            names = set().union(*(frame["host_ms"] for frame in group))
            cold_routes, chip_calls, stream_totals, worker_critical = [], [], [], []
            for frame in group:
                if [row["layer"] for row in frame["dispatches"]] != list(range(18, 33)):
                    raise ValueError("incomplete middle-layer profile")
                if not math.isclose(sum(frame["host_ms"].values()), frame["total_ms"], abs_tol=1e-6):
                    raise ValueError("host spans do not account for full forward time")
                if any(not math.isfinite(value) or value < -1e-6 for value in frame["host_ms"].values()):
                    raise ValueError("invalid host span")
                cold_routes.append(sum(sum(dispatch["cold_routes"].values()) for dispatch in frame["dispatches"]))
                chip_calls.append(
                    sum(
                        sum("header" in event for event in dispatch["socket_events"])
                        for dispatch in frame["dispatches"]
                    )
                )
                device = {}
                for event in frame["device_stream_ms"]:
                    device[event["name"]] = device.get(event["name"], 0) + event["ms"]
                stream_totals.append(device)
                total = {"upload": 0, "native": 0, "download": 0, "all_stages": 0}
                for dispatch in frame["dispatches"]:
                    called = []
                    for event in dispatch["socket_events"]:
                        if "header" not in event:
                            continue
                        sequence, layer, rows = event["header"]
                        if layer != dispatch["layer"] or rows != dispatch["rows"]:
                            raise ValueError("native profile header mismatch")
                        key = event["rank"], sequence, layer, rows
                        if key not in by_invocation:
                            raise ValueError("missing native profile invocation")
                        called.append(by_invocation[key]["stream_ms"])
                    if called:
                        for name in ("upload", "native", "download"):
                            total[name] += max(row[name] for row in called)
                        total["all_stages"] += max(sum(row.values()) for row in called)
                worker_critical.append(total)
            means = {name: statistics.mean(frame["host_ms"].get(name, 0) for frame in group) for name in sorted(names)}
            result["phases"][phase] = dict(
                steps=len(group),
                tokens=sum(frame["rows"] for frame in group),
                mean_forward_ms=statistics.mean(frame["total_ms"] for frame in group),
                median_forward_ms=statistics.median(frame["total_ms"] for frame in group),
                host_mean_ms=means,
                host_share_percent={name: 100 * value / sum(means.values()) for name, value in means.items()},
                mean_cold_route_fraction=sum(cold_routes) / (sum(frame["rows"] for frame in group) * 15 * 8),
                mean_chip_calls_per_step=statistics.mean(chip_calls),
                middle_backbone_per_layer_stream_mean_ms={
                    str(layer): statistics.mean(
                        event["ms"]
                        for frame in group
                        for event in frame["device_stream_ms"]
                        if event["name"] == "middle_backbone" and event["layer"] == layer
                    )
                    for layer in range(18, 33)
                    if any(
                        event["name"] == "middle_backbone" and event.get("layer") == layer
                        for event in group[0]["device_stream_ms"]
                    )
                },
                cuda_stream_mean_ms={
                    name: statistics.mean(row.get(name, 0) for row in stream_totals)
                    for name in set().union(*stream_totals)
                },
                npu_sum_of_layer_rank_maxima_mean_ms={
                    name: statistics.mean(row[name] for row in worker_critical) for name in worker_critical[0]
                },
            )
        report["workloads"].append(result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads((args.root / "profile-run.json").read_bytes()),
        read_rows(args.root / "parent-profile.jsonl.gz"),
        read_rows(args.root / "worker-profile-rank0.jsonl.gz") + read_rows(args.root / "worker-profile-rank1.jsonl.gz"),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
