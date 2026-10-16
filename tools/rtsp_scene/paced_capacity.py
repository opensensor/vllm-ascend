"""Replay consecutive camera windows at a fixed arrival rate and retain backlog."""

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path

from tools.rtsp_scene.multi_stream import has_budget
from tools.rtsp_scene.runner import SCENE_PROMPT, describe, safe_error


def arrival_offset(window: int, interval: float) -> float:
    if window < 0 or not math.isfinite(interval) or not 0 < interval <= 60:
        raise ValueError("Use nonnegative window indices and an interval in (0, 60] seconds")
    return window * interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--interval", type=float, default=8)
    parser.add_argument("--max-in-flight", type=int, default=16)
    parser.add_argument("--endpoint", action="append", required=True)
    args = parser.parse_args()
    if min(args.streams, args.windows, args.max_in_flight) <= 0:
        parser.error("Counts must be positive")
    try:
        arrival_offset(0, args.interval)
    except ValueError as error:
        parser.error(str(error))
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [(directory, json.loads((directory / "capture.json").read_text())) for directory in args.captures]
    started = time.monotonic()
    jobs, records, arrivals = [], [], []

    def run_request(directory, manifest, request, record):
        response = describe(directory, manifest, request, record)
        return response, time.monotonic() - started

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_in_flight) as pool:
        for window in range(args.windows):
            target = started + arrival_offset(window, args.interval)
            time.sleep(max(0, target - time.monotonic()))
            pending = sum(not job.done() for job in jobs)
            arrivals.append({"window": window, "seconds": time.monotonic() - started, "earlier_pending": pending})
            print(json.dumps(arrivals[-1]), flush=True)
            for stream in range(args.streams):
                record = {"stream": stream, "window": window, "arrival_seconds": time.monotonic() - started}
                if not has_budget(jobs, args.max_in_flight):
                    record["dropped"] = "Inference budget exhausted"
                else:
                    source, manifest = sources[(stream + window) % len(sources)]
                    output = args.output / f"s{stream}-w{window}"
                    output.mkdir()
                    prompt = output / "request-prompt.txt"
                    prompt.write_text("Request identifier: " + uuid.uuid4().hex + "\n" + SCENE_PROMPT)
                    request = argparse.Namespace(
                        model="qwen3-vl-scene", endpoint=args.endpoint[stream % len(args.endpoint)], prompt_file=prompt
                    )
                    record["endpoint"] = request.endpoint
                    record["capture"] = str(source)
                    record["job_index"] = len(jobs)
                    jobs.append(pool.submit(run_request, source, manifest, request, output))
                records.append(record)
        for record in records:
            if "job_index" not in record:
                continue
            try:
                response, finished = jobs[record["job_index"]].result()
                record["finished_seconds"] = finished
                record["response_seconds"] = finished - record["arrival_seconds"]
                record["completion_tokens"] = response["usage"]["completion_tokens"]
            except Exception as error:
                record["error"] = safe_error(str(error))
    latencies = [record["response_seconds"] for record in records if "response_seconds" in record]
    summary = {
        "streams": args.streams,
        "windows_per_stream": args.windows,
        "arrival_interval_seconds": args.interval,
        "elapsed_seconds": time.monotonic() - started,
        "completed": len(latencies),
        "dropped": sum("dropped" in record for record in records),
        "failed": sum("error" in record for record in records),
        "median_response_seconds": statistics.median(latencies) if latencies else None,
        "max_response_seconds": max(latencies) if latencies else None,
        "responses_longer_than_interval": sum(value > args.interval for value in latencies),
        "arrivals": arrivals,
        "requests": records,
        "note": (
            "Finite paced replay; arrival is fixed independently of inference. No live camera transport is involved."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"arrivals", "requests"}}), flush=True)
    if summary["failed"]:
        raise RuntimeError("One or more replay requests failed; inspect summary.json")


if __name__ == "__main__":
    main()
