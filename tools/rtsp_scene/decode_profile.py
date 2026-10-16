"""Bounded text-only profiles at realistic scene KV lengths; no capacity claims."""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_PROMPT_TOKENS = 2244
DEFAULT_OUTPUT_TOKENS = 16
ANALYSIS_PROCESSES = 4


def profile_body(prompt_tokens, output_tokens, token):
    if prompt_tokens <= 0 or output_tokens < 2 or token < 0:
        raise ValueError("Use a positive context, at least two output tokens and a valid token")
    # The pinned server accepts token-ID prompts. Repeating a valid text token
    # controls KV length without including vision encoder work in the trace.
    return {
        "model": "qwen3-vl-scene",
        "prompt": [token] * prompt_tokens,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8003")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--token", type=int, default=1234)
    args = parser.parse_args()
    if args.output.exists() or not args.concurrency or min(args.concurrency) < 1:
        parser.error("Use a fresh private directory and positive concurrency")
    body = profile_body(args.prompt_tokens, args.output_tokens, args.token)
    args.output.mkdir(parents=True, mode=0o700)

    def post(route, payload=None):
        request = urllib.request.Request(
            args.server + route,
            data=json.dumps(payload).encode() if payload is not None else b"",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def send():
        result = post("/v1/completions", body)
        if result["usage"]["prompt_tokens"] != args.prompt_tokens:
            raise ValueError("Server context length does not match the profile contract")
        if result["usage"]["completion_tokens"] != args.output_tokens:
            raise ValueError("Server did not finish the diagnostic output budget")
        return {"usage": result["usage"], "finish_reason": result["choices"][0]["finish_reason"]}

    records = []
    for concurrency in args.concurrency:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda _: send(), range(concurrency)))
        previous = set(args.profile_root.glob("*"))
        post("/start_profile")
        try:
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(lambda _: send(), range(concurrency)))
            elapsed = time.monotonic() - started
        finally:
            post("/stop_profile")
        # Workers are daemon processes and cannot run the profiler's process
        # pool. Parse collected traces explicitly in a non-daemon subprocess.
        deadline = time.monotonic() + 120
        while True:
            created = [p for p in args.profile_root.glob("*") if p not in previous and p.is_dir()]
            if len(created) >= 2:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Profiler did not produce both worker traces")
            time.sleep(1)
        for directory in created:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from torch_npu.profiler.profiler import analyse; "
                    "analyse(sys.argv[1], max_process_number=int(sys.argv[2]))",
                    str(directory),
                    str(ANALYSIS_PROCESSES),
                ],
                check=True,
                timeout=240,
            )
        csvs = [p for directory in created for p in directory.rglob("kernel_details.csv")]
        if len(csvs) < 2:
            raise ValueError("Offline parser did not produce both worker kernel CSVs")
        destination = args.output / f"c{concurrency}"
        destination.mkdir(mode=0o700)
        for directory in created:
            shutil.copytree(directory, destination / directory.name)
        records.append(
            {
                "concurrency": concurrency,
                "context_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
                "profiled_wall_seconds": elapsed,
                "requests": results,
                "traces": [str(destination / p.name) for p in created],
                "scope": "Text-only matching scene KV length; excludes vision; profiler overhead present",
                "diagnostic_only": True,
            }
        )
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({k: v for k, v in records[-1].items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
