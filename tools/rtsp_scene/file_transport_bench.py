"""Pair base64/file requests with identical frames, prompts and fixed token budgets."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from tools.rtsp_scene.fixed_decode import request_body
from tools.rtsp_scene.runner import API_TIMEOUT_SECONDS, SCENE_PROMPT


def group_cpu(pids):
    total = 0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        total += int(fields[11]) + int(fields[12])
    return total / os.sysconf("SC_CLK_TCK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8003/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--server-pids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.output.exists() or min(*args.concurrency, args.rounds, args.tokens) <= 0:
        parser.error("Use a fresh output and positive counts")
    if urlparse(args.endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("File transport requires a loopback model endpoint")
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700)
    sources = []
    for capture in args.captures:
        manifest = json.loads((capture / "capture.json").read_text())
        sources.append(
            {
                "capture": capture,
                "frame_sha256": [
                    hashlib.sha256((capture / frame["file"]).read_bytes()).hexdigest() for frame in manifest["frames"]
                ],
            }
        )
    results = []

    def send(index, prompt, transport):
        source = sources[index % len(sources)]
        prepared = time.monotonic()
        body = request_body(source["capture"], prompt, args.tokens, transport, args.frame_root)
        preparation = time.monotonic() - prepared
        started = time.monotonic()
        request = urllib.request.Request(args.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
        if payload["usage"]["completion_tokens"] != args.tokens:
            raise ValueError("Fixed token budget differs")
        return {
            "index": index,
            "frame_sha256": source["frame_sha256"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "usage": payload["usage"],
            "preparation_seconds": preparation,
            "request_bytes": len(body),
            "response_seconds": time.monotonic() - started,
        }

    # Warm both media loaders; exclude these requests from recorded comparisons.
    for transport in ("base64", "file"):
        send(0, "Transport warmup\n" + SCENE_PROMPT, transport)
    for concurrency in args.concurrency:
        for iteration in range(args.rounds):
            prompt = f"Transport cohort {concurrency}, iteration {iteration}\n" + SCENE_PROMPT
            for transport in ("base64", "file") if iteration % 2 == 0 else ("file", "base64"):
                cpu, server_cpu, wall = time.process_time(), group_cpu(args.server_pids), time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    requests = list(
                        pool.map(
                            lambda index, prompt=prompt, transport=transport: send(index, prompt, transport),
                            range(concurrency),
                        )
                    )
                record = {
                    "concurrency": concurrency,
                    "iteration": iteration,
                    "transport": transport,
                    "batch_seconds": time.monotonic() - wall,
                    "client_cpu_seconds": time.process_time() - cpu,
                    "server_group_cpu_seconds": group_cpu(args.server_pids) - server_cpu,
                    "requests": requests,
                }
                results.append(record)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k != "requests"}), flush=True)
    for left, right in zip(results[::2], results[1::2]):
        for a, b in zip(left["requests"], right["requests"]):
            for key in ("index", "frame_sha256", "prompt_sha256", "usage"):
                if a[key] != b[key]:
                    raise ValueError(f"Transport pair differs in {key}")
    (args.output / "gate.json").write_text(
        json.dumps(
            {
                "passed": True,
                "matched_pairs": len(results) // 2,
                "note": "Fixed-token truncated diagnostic; server CPU includes overlapped work and busy waits.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
