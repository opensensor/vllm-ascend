"""Compare fixed-token serving runs only when their request workloads match."""

import argparse
import json
import statistics
from pathlib import Path


def compare(control: Path, candidate: Path) -> dict:
    before = json.loads((control / "summary.json").read_text())
    after = json.loads((candidate / "summary.json").read_text())
    if [group["concurrency"] for group in before] != [group["concurrency"] for group in after]:
        raise ValueError("Concurrency cohorts differ")
    results = []
    for reference, trial in zip(before, after):
        concurrency = reference["concurrency"]
        if not reference.get("diagnostic_only") or not trial.get("diagnostic_only"):
            raise ValueError("Use fixed-token diagnostic runs")
        if reference["fixed_completion_tokens"] != trial["fixed_completion_tokens"]:
            raise ValueError("Fixed token budgets differ")
        if reference["requests"] != trial["requests"] or len(reference["batch_seconds"]) != len(trial["batch_seconds"]):
            raise ValueError("Request counts differ")
        matched = 0
        for iteration in range(len(reference["batch_seconds"])):
            for index in range(concurrency):
                name = f"c{concurrency}-r{iteration}-{index}.json"
                records = [json.loads((directory / name).read_text()) for directory in (control, candidate)]
                left, right = records
                if left["body_sha256"] != right["body_sha256"] or left["usage"] != right["usage"]:
                    raise ValueError(f"Request body or token workload differs: {name}")
                if any(
                    record["usage"]["completion_tokens"] != reference["fixed_completion_tokens"] for record in records
                ):
                    raise ValueError(f"Fixed token budget differs: {name}")
                matched += 1
        if matched != reference["requests"]:
            raise ValueError("Recorded requests do not match the summary")
        timing = {}
        for metric in ("request_prefill_time_seconds", "request_decode_time_seconds"):
            baseline = reference["server_mean_seconds"][metric]
            optimized = trial["server_mean_seconds"][metric]
            if baseline is None or optimized is None or min(baseline, optimized) <= 0:
                raise ValueError("Positive server timings are required")
            timing[metric] = {
                "control": baseline,
                "candidate": optimized,
                "time_reduction_percent": 100 * (1 - optimized / baseline),
            }
        baseline = statistics.mean(reference["batch_seconds"])
        optimized = statistics.mean(trial["batch_seconds"])
        results.append(
            {
                "concurrency": concurrency,
                "matched_requests": matched,
                "fixed_completion_tokens": reference["fixed_completion_tokens"],
                "server_mean_seconds": timing,
                "batch_mean_seconds": {"control": baseline, "candidate": optimized},
                "batch_time_reduction_percent": 100 * (1 - optimized / baseline),
            }
        )
    return {
        "control": str(control),
        "candidate": str(candidate),
        "results": results,
        "limitations": (
            "Finite fixed-token diagnostic: outputs are truncated and are not scene reports. "
            "Results do not establish sustained camera capacity or description accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.control, args.candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
