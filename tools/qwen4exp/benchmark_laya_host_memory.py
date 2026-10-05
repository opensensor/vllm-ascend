# SPDX-License-Identifier: Apache-2.0
"""Measure a CPU-only Laya decision model's resident memory and latency.

This is a host probe. Run in a clean CPU-only Python environment. It neither
imports torch_npu nor starts a model server.
"""

import argparse
import importlib
import json
import platform
import statistics
import time
from pathlib import Path

import psutil

DEFAULT_MODEL_ID = "convaiinnovations/laya"
QUESTION = {
    "category": {
        "type": "choice",
        "instructions": "Which kind of request is this?",
        "criteria": {
            "technical": "a software failure or bug report",
            "billing": "a payment, invoice, or refund request",
            "other": "a request about another subject",
        },
    },
}
SHORT_STATE = "The inference server failed to start after a package update. Please inspect the error log."
LONG_STATE = " ".join([SHORT_STATE] * 20)


def memory_mib() -> dict[str, float]:
    info = psutil.Process().memory_full_info()
    return {
        "rss": round(info.rss / (1024 * 1024), 1),
        "uss": round(info.uss / (1024 * 1024), 1),
    }


def measure(agent, state: str, runs: int) -> dict[str, object]:
    agent.predict(state, QUESTION)
    times = []
    answer = None
    for _ in range(runs):
        start = time.perf_counter()
        answer = agent.predict(state, QUESTION)["answers"]["category"]["choice"]
        times.append(1000 * (time.perf_counter() - start))
    return {
        "median_ms": round(statistics.median(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "trials_ms": [round(value, 2) for value in times],
        "answer": answer,
    }


def measure_batch(agent, decisions: int, runs: int) -> dict[str, object]:
    questions = {
        f"decision_{index}": {
            **QUESTION["category"],
            "instructions": f"Which kind of request is aspect {index + 1}?",
        }
        for index in range(decisions)
    }
    agent.predict(SHORT_STATE, questions)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        answers = agent.predict(SHORT_STATE, questions)["answers"]
        times.append(1000 * (time.perf_counter() - start))
        if len(answers) != decisions:
            raise RuntimeError("Laya returned fewer decisions than requested")
    median = statistics.median(times)
    return {
        "median_ms": round(median, 2),
        "per_decision_ms": round(median / decisions, 2),
        "trials_ms": [round(value, 2) for value in times],
        "after_inference_mib": memory_mib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--subfolder")
    parser.add_argument("--revision", required=True, help="Pinned Hub commit for the model")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs <= 0 or any(count <= 0 for count in (*args.threads, *args.batch_sizes)):
        parser.error("runs, thread counts, and batch sizes must be positive")

    record = {
        "host": platform.node(),
        "cpu": platform.processor(),
        "model": args.model_id,
        "subfolder": args.subfolder,
        "revision": args.revision,
        "baseline_mib": memory_mib(),
    }
    # Import cost belongs in the memory budget, so import these after baseline.
    torch = importlib.import_module("torch")
    laya = importlib.import_module("laya")
    record["torch_version"] = torch.__version__
    record["laya_version"] = laya.__version__
    record["after_import_mib"] = memory_mib()
    torch.set_num_threads(args.threads[0])

    start = time.perf_counter()
    agent = laya.load(args.model_id, device="cpu", subfolder=args.subfolder, revision=args.revision)
    record["load_seconds"] = round(time.perf_counter() - start, 2)
    record["after_load_mib"] = memory_mib()
    record["cases"] = {}
    for threads in args.threads:
        torch.set_num_threads(threads)
        record["cases"][str(threads)] = {
            "short": measure(agent, SHORT_STATE, args.runs),
            "long": measure(agent, LONG_STATE, args.runs),
            "after_inference_mib": memory_mib(),
        }
        if args.batch_sizes:
            record["cases"][str(threads)]["batched_short"] = {
                str(count): measure_batch(agent, count, args.runs) for count in args.batch_sizes
            }
    record["final_mib"] = memory_mib()
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
