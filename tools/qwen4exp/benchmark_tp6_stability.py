# SPDX-License-Identifier: Apache-2.0
"""Exercise prefill allocation churn, prefix reuse and idle-to-decode recovery.

This is a bounded serving stress check, not a long-context accuracy benchmark.
Every request has a deadline; HTTP health alone is never treated as progress.
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def completion(url, model, prompt, tokens, timeout):
    payload = dict(
        model=model,
        messages=[dict(role="user", content=prompt)],
        temperature=0,
        max_tokens=tokens,
        stream=True,
        stream_options=dict(include_usage=True),
    )
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    arrivals, output, usage = [], [], None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if time.monotonic() - started > timeout:
                raise TimeoutError("completion exceeded total request deadline")
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if text:
                    arrivals.append(time.monotonic())
                    output.append(text)
    if not done or not output or not usage or usage.get("completion_tokens", 0) < 1:
        raise RuntimeError("request did not finish with generated content and token usage")
    return dict(
        elapsed_s=time.monotonic() - started,
        ttft_s=arrivals[0] - started,
        max_chunk_gap_s=max((b - a for a, b in zip(arrivals, arrivals[1:])), default=0),
        usage=usage,
        text="".join(output),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen38-flash-next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.rounds < 1 or args.idle_seconds < 0:
        parser.error("positive round count and nonnegative idle duration required")
    # Exclusive creation preserves a partial record when a request times out.
    with args.output.open("x") as log:

        def run(label, prompt, tokens=256):
            result = completion(args.url, args.model, prompt, tokens, 240)
            log.write(json.dumps(dict(label=label, **result)) + "\n")
            log.flush()
            print(label, result["usage"], result["elapsed_s"], flush=True)

        for cycle in range(args.rounds):
            prefix = f"Session {time.time_ns()}-{cycle}. Reference notes:\n" + "alpha beta gamma delta " * 5600
            run(f"cold-{cycle}", prefix + "\nExplain how to debug a deadlock in a multithreaded program.")
            run(
                f"prefix-{cycle}",
                prefix + "epsilon zeta eta theta " * 1000 + "\nDescribe a careful test plan for the fix.",
            )
            time.sleep(args.idle_seconds)
            run(f"resume-{cycle}", "What is 19 plus 23? Answer briefly.", 32)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(
                    completion,
                    args.url,
                    args.model,
                    f"Case {i}: explain mutexes and event queues with an example.",
                    128,
                    240,
                )
                for i in range(6)
            ]
            for i, future in enumerate(futures):
                log.write(json.dumps(dict(label=f"six-{i}", **future.result())) + "\n")
                log.flush()
        log.write(json.dumps(dict(passed=True, rounds=args.rounds, six_requests=6)) + "\n")


if __name__ == "__main__":
    main()
