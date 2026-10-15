# SPDX-License-Identifier: Apache-2.0
"""Compare matched QFN receipts without treating throughput as quality proof."""

import argparse
import json
import math
import statistics
from pathlib import Path

from .qfn_collective_benchmark import expects_graph_replay, graph_sizes, validate_replay_evidence
from .qfn_collective_worker import validate_graph_snapshot


def prefix_length(first, second):
    return next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))


def compare(baseline, candidate, baseline_options, candidate_options):
    options = dict(candidate_options)
    additional = options.pop("additional_config", {})
    if "npu_collective" not in additional:
        raise ValueError("candidate options do not enable the custom collective")
    if options != baseline_options:
        raise ValueError("engine options differ beyond the custom collective")
    profile = baseline.get("graph_profile", "eager")
    sizes = graph_sizes(profile)
    if candidate.get("graph_profile", "eager") != profile:
        raise ValueError("graph profiles differ")
    eager = not bool(sizes)
    if baseline_options.get("enforce_eager") is not eager:
        raise ValueError("engine eager option differs from graph profile")
    if sizes and baseline_options.get("compilation_config") != dict(
        mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=list(sizes)
    ):
        raise ValueError("graph compilation options differ from the qualified profile")
    for result, mode in ((baseline, "hccl"), (candidate, "custom")):
        if (
            result.get("mode") != mode
            or result.get("eager") is not eager
            or result.get("ownership_released") is not True
        ):
            raise ValueError("completed matched-profile receipts with released ownership are required")
    if sizes:
        workers = candidate.get("closed_workers")
        if type(workers) is not list or len(workers) != 4:
            raise ValueError("all four graph retirement receipts are required")
        for worker in workers:
            snapshot = validate_graph_snapshot(worker.get("graph_observer"))
            if (
                worker.get("state") != "closed"
                or snapshot.get("retired") is not True
                or snapshot.get("active_capture") is not False
                or snapshot.get("lifecycle_only") is not True
                or snapshot["failed_graphs"] != 0
                or snapshot["captured_graphs"] <= 0
                or type(snapshot.get("registered_graphs")) is not int
                or snapshot["registered_graphs"] <= 0
                or type(snapshot.get("reset_graphs")) is not int
                or snapshot["reset_graphs"] != snapshot["registered_graphs"]
                or snapshot["native_leases_acquired"] != snapshot["native_leases_released"]
            ):
                raise ValueError("custom graphs must retire before native owner release")
    first, second = baseline["measurements"], candidate["measurements"]
    if len(first) != len(second):
        raise ValueError("measurement coverage differs")
    trials = []
    for a, b in zip(first, second, strict=True):
        for key in ("label", "concurrency", "generated_tokens"):
            if a[key] != b[key]:
                raise ValueError("workload differs: " + key)
        for trial in (a, b):
            duration = trial["elapsed_seconds"]
            if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("positive finite trial duration required")
            if len(trial["outputs"]) != trial["concurrency"]:
                raise ValueError("request count differs from concurrency")
        requests = []
        for aa, bb in zip(a["outputs"], b["outputs"], strict=True):
            for key in ("prompt_tokens", "num_cached_tokens"):
                if aa[key] != bb[key]:
                    raise ValueError("prompt/cache workload differs: " + key)
            if len(aa["completions"]) != 1 or len(bb["completions"]) != 1:
                raise ValueError("single completion receipts required")
            av, bv = aa["completions"][0]["token_ids"], bb["completions"][0]["token_ids"]
            if len(av) != len(bv):
                raise ValueError("completion token budgets differ")
            requests.append(
                dict(
                    tokens=len(av),
                    equal_token_ids=av == bv,
                    shared_prefix_tokens=prefix_length(av, bv),
                    num_cached_tokens=aa["num_cached_tokens"],
                )
            )
        coverage = b["collective_coverage"]
        if len(coverage) != 4 or not sizes and any(not sum(g["calls"] for g in r["geometries"]) for r in coverage):
            raise ValueError("custom execution must be proved on all four ranks per trial")
        graph_coverage = None
        if sizes:
            required = expects_graph_replay(profile, a["concurrency"])
            if any(trial.get("expected_graph_replay") is not required for trial in (a, b)):
                raise ValueError("trial graph expectation differs from configured capture sizes")
            graph_coverage = b.get("graph_replay_coverage")
            validate_replay_evidence(graph_coverage, required=required)
            tracking = additional["npu_collective"].get("graph_replay_tracking", False)
            if type(tracking) is not bool or any(rank["tracking_enabled"] is not tracking for rank in graph_coverage):
                raise ValueError("graph observer tracking differs from configured production/diagnostic mode")
        rates = [trial["generated_tokens"] / trial["elapsed_seconds"] for trial in (a, b)]
        trials.append(
            dict(
                label=a["label"],
                concurrency=a["concurrency"],
                hccl_tokens_per_second=rates[0],
                custom_tokens_per_second=rates[1],
                speed_ratio=rates[1] / rates[0],
                requests=requests,
                custom_calls_per_rank=[sum(g["calls"] for g in r["geometries"]) for r in coverage],
                fallback_calls_per_rank=[sum(g["calls"] for g in r["fallback_schemas"]) for r in coverage],
            )
        )
        if sizes:
            trials[-1]["expected_graph_replay"] = required
            trials[-1]["graph_replay_coverage"] = graph_coverage
            trials[-1]["replay_counter_observed"] = graph_coverage[0]["tracking_enabled"]
    summaries = []
    for concurrency in sorted({t["concurrency"] for t in trials if t["label"] != "warmup"}):
        selected = [t for t in trials if t["label"] != "warmup" and t["concurrency"] == concurrency]
        base = statistics.median(t["hccl_tokens_per_second"] for t in selected)
        new = statistics.median(t["custom_tokens_per_second"] for t in selected)
        summaries.append(
            dict(
                concurrency=concurrency,
                trials=len(selected),
                hccl_median_tokens_per_second=base,
                custom_median_tokens_per_second=new,
                median_rate_ratio=new / base,
            )
        )
    return dict(
        engine_options_matched=True,
        ownership_released=True,
        eager=eager,
        graph_profile=profile,
        graph_evidence_scope=(
            "cold custom captures/native leases; production replay counts unobserved; "
            "graph selection requires qualified wrapper/runtime dispatch log evidence; "
            "baseline capture configuration matched"
        ),
        trials=trials,
        summaries=summaries,
        confidence_interval_computed=False,
        model_numerical_qualification=False,
        scope="completed generation including prefill; short prompts; two timed trials per concurrency",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-options", "candidate-options", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = compare(
        *[
            json.loads(path.read_text())
            for path in (args.baseline, args.candidate, args.baseline_options, args.candidate_options)
        ]
    )
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summaries"]))


if __name__ == "__main__":
    main()
