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
    decode_only = run.get("decode_only", False)
    if decode_only and not all(row.get("full_vocabulary_bitwise_equal") for row in run["workloads"]):
        raise ValueError("decode profile requires parity for every measured workload")
    direct = run.get("transport") == "direct-dma"
    stage_names = ("upload", "validation", "native", "download") if direct else ("upload", "native", "download")
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
        worker_timing_scope="host wall including completion waits" if direct else "device stream",
        workloads=[],
    )
    for workload in run["workloads"]:
        selected = [
            frame for frame in frames if workload["first_sequence"] <= frame["sequence"] < workload["end_sequence"]
        ]
        result = dict(name=workload["name"], cuda_kernel_capture=workload.get("cuda_kernel_capture", False), phases={})
        for phase in ("decode",) if decode_only else ("prefill", "decode"):
            group = [
                frame for frame in selected if (frame["position"] < workload["prompt_tokens"]) == (phase == "prefill")
            ]
            if not group:
                raise ValueError("missing observed prefill or decode")
            if decode_only and (
                len(group) != workload["output_tokens"] - 1
                or any(frame["rows"] != 1 for frame in group)
                or len({frame["sequence"] for frame in group}) != len(group)
            ):
                raise ValueError("decode profile must observe every single-token forward exactly once")
            names = set().union(*(frame["host_ms"] for frame in group))
            cold_routes, chip_calls, stream_totals, worker_critical = [], [], [], []
            for frame in group:
                if [row["layer"] for row in frame["dispatches"]] != list(range(18, 33)):
                    raise ValueError("incomplete middle-layer profile")
                if not math.isclose(sum(frame["host_ms"].values()), frame["total_ms"], abs_tol=1e-6):
                    raise ValueError("host spans do not account for full forward time")
                if any(not math.isfinite(value) or value < -1e-6 for value in frame["host_ms"].values()):
                    raise ValueError("invalid host span")
                if all(dispatch["cold_routes"] is not None for dispatch in frame["dispatches"]):
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
                total = dict.fromkeys((*stage_names, "all_stages"), 0)
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
                        worker = by_invocation[key]
                        called.append(
                            dict(
                                upload=worker["upload_ms"],
                                validation=worker["unpack_and_validation_ms"],
                                native=worker["native_ms"],
                                download=worker["pack_and_download_ms"],
                            )
                            if direct
                            else worker["stream_ms"]
                        )
                    if called:
                        for name in stage_names:
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
                mean_cold_route_fraction=(
                    sum(cold_routes) / (sum(frame["rows"] for frame in group) * 15 * 8)
                    if len(cold_routes) == len(group)
                    else None
                ),
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
    run = json.loads((args.root / "profile-run.json").read_bytes())
    if run.get("transport") == "direct-dma":
        workers = []
        for rank in (0, 1):
            receipt = json.loads((args.root / f"dma-worker-rank{rank}.json").read_bytes())
            workers.extend(dict(row, rank=rank) for row in receipt["profile"])
    else:
        workers = read_rows(args.root / "worker-profile-rank0.jsonl.gz") + read_rows(
            args.root / "worker-profile-rank1.jsonl.gz"
        )
    frames = args.root / "parent-profile.jsonl"
    result = summarize(run, read_rows(frames if frames.exists() else frames.with_suffix(".jsonl.gz")), workers)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
