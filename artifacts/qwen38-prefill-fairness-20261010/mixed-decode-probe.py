# SPDX-License-Identifier: Apache-2.0
"""Bounded live fairness probe; explicit --execute and idle admission required."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--thermal-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print("Dry run: one 384-token decoder and one roughly 2K-token prefill; no requests submitted")
        return
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in response.read().decode().splitlines()
            if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
        ]
    assert len(values) == 2 and all(value == 0 for value in values), "User traffic; defer probe"
    ready = threading.Event()
    handles = {}
    lock = threading.Lock()
    nonce = str(time.time_ns())

    def stream(name, prompt, maximum):
        body = {
            "model": "qwen38-flash-next",
            "prompt": prompt,
            "max_tokens": maximum,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            base + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        times = []
        usage = None
        with urllib.request.urlopen(request, timeout=180) as response:
            with lock:
                handles[name] = response
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                if any(choice.get("text") for choice in event.get("choices", [])):
                    times.append(time.monotonic())
                    if name == "decode" and len(times) >= 3:
                        ready.set()
        with lock:
            handles.pop(name, None)
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        return {
            "name": name,
            "started_monotonic": started,
            "ended_monotonic": time.monotonic(),
            "ttft_s": times[0] - started if times else None,
            "content_event_times": times,
            "gap_p50_s": gaps[len(gaps) // 2] if gaps else None,
            "gap_p95_s": gaps[min(len(gaps) - 1, int(len(gaps) * 0.95))] if gaps else None,
            "gap_max_s": max(gaps) if gaps else None,
            "usage": usage,
        }

    temperatures = []
    error = None
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decoder = pool.submit(
            stream, "decode", "Fairness probe " + nonce + ": Write a detailed numbered Python testing guide.", 384
        )
        assert ready.wait(30), "Decoder did not begin streaming"
        prompt = (
            "Cold prefill probe "
            + nonce
            + ":\n"
            + "The archive describes a blue project with tests and a completed review.\n" * 160
        )
        prefill = pool.submit(stream, "prefill", prompt + "Continue the archive briefly.", 16)
        while not (decoder.done() and prefill.done()):
            thermal = json.loads(args.thermal_log.read_text().splitlines()[-1])
            maximum = thermal.get("maximum_c")
            temperatures.append({"time": time.time(), "maximum_c": maximum})
            if maximum is None or maximum >= 90 or time.time() - thermal["time"] > 10:
                error = "Controlled thermal abort or stale sensor"
                with lock:
                    for handle in handles.values():
                        handle.close()
                break
            time.sleep(1)
        for future in (decoder, prefill):
            try:
                results.append(future.result())
            except Exception as exception:
                error = error or f"{type(exception).__name__}: {exception}"
    receipt = {
        "results": results,
        "temperatures": temperatures,
        "error": error,
        "timing_unit": "SSE content events; one event may contain multiple accepted tokens",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "error": error,
                "peak_c": max(value["maximum_c"] or 0 for value in temperatures),
                "results": [
                    {key: value for key, value in result.items() if key != "content_event_times"} for result in results
                ],
            }
        )
    )
    assert error is None and all(result["usage"] for result in results)


if __name__ == "__main__":
    main()
