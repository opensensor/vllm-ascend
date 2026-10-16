"""Measure bounded concurrent scene requests, including server queue and prefill time."""

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

from tools.rtsp_scene.runner import SCENE_PROMPT, describe


def metrics(endpoint: str) -> dict[str, float]:
    address = endpoint.removesuffix("/v1/chat/completions") + "/metrics"
    with urllib.request.urlopen(address, timeout=5) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for line in lines:
        if line.startswith("vllm:") and ("_sum{" in line or "_count{" in line):
            name, value = line.rsplit(" ", 1)
            name = name.split("{", 1)[0].removeprefix("vllm:")
            values[name] = values.get(name, 0) + float(value)
    return values


def aggregate_metrics(endpoints: list[str]) -> dict[str, float]:
    values = {}
    for endpoint in endpoints:
        for key, value in metrics(endpoint).items():
            values[key] = values.get(key, 0) + value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--endpoint", action="append", help="Repeat to distribute requests across model replicas")
    parser.add_argument("--model", default="qwen3-vl-scene")
    parser.add_argument("--reference-prompts", type=Path, help="Reuse exact prompts from an earlier capacity run")
    args = parser.parse_args()
    args.endpoint = args.endpoint or ["http://127.0.0.1:8003/v1/chat/completions"]
    if args.rounds <= 0 or any(value <= 0 for value in args.concurrency):
        parser.error("Rounds and concurrency must be positive")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for concurrency in args.concurrency:
        before_metrics = aggregate_metrics(args.endpoint)
        batches, latencies, completions = [], [], []
        for iteration in range(args.rounds):
            jobs = []
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for index in range(concurrency):
                    directory = args.captures[index % len(args.captures)]
                    record = args.output / f"c{concurrency}-r{iteration}-{index}"
                    record.mkdir()
                    # Unique text before images avoids reuse of an earlier KV prefix.
                    prompt_file = record / "request-prompt.txt"
                    if args.reference_prompts:
                        text = (args.reference_prompts / record.name / "request-prompt.txt").read_text()
                    else:
                        text = "Request identifier: " + uuid.uuid4().hex + "\n" + SCENE_PROMPT
                    prompt_file.write_text(text)
                    request_args = argparse.Namespace(
                        model=args.model, endpoint=args.endpoint[index % len(args.endpoint)], prompt_file=prompt_file
                    )
                    manifest = json.loads((directory / "capture.json").read_text())
                    jobs.append(pool.submit(describe, directory, manifest, request_args, record))
                for job in jobs:
                    result = job.result()
                    latencies.append(result["latency_seconds"])
                    completions.append(result["usage"]["completion_tokens"])
            batches.append(time.monotonic() - started)
        after_metrics = aggregate_metrics(args.endpoint)
        server_means = {}
        for name in [
            "request_queue_time_seconds",
            "request_prefill_time_seconds",
            "request_decode_time_seconds",
            "time_to_first_token_seconds",
        ]:
            count = after_metrics.get(name + "_count", 0) - before_metrics.get(name + "_count", 0)
            total = after_metrics.get(name + "_sum", 0) - before_metrics.get(name + "_sum", 0)
            server_means[name] = total / count if count else None
        summary = {
            "endpoints": args.endpoint,
            "captures": [str(directory) for directory in args.captures],
            "reference_prompts": str(args.reference_prompts) if args.reference_prompts else None,
            "concurrency": concurrency,
            "requests": len(latencies),
            "batch_seconds": batches,
            "median_client_seconds": statistics.median(latencies),
            "max_client_seconds": max(latencies),
            "windows_per_second": len(latencies) / sum(batches),
            "completion_tokens": completions,
            "server_mean_seconds": server_means,
            "note": "Replayed frames; client time includes queueing. Not a sustained live-stream test.",
        }
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
