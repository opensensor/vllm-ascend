# SPDX-License-Identifier: Apache-2.0
"""Serial streaming decode A/B, with per-request output and acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = (
    "Write a detailed Python implementation of an LRU cache with unit tests. Explain thread safety and complexity.",
    "Review a producer-consumer queue design. Explain bounded capacity, cancellation, shutdown "
    "and deadlock prevention. "
    "Then implement it in Python and give tests for failure cases.",
    "Write a practical guide to investigating a slow distributed inference service. Include concrete experiments, "
    "how to isolate CPU, memory, communication and accelerator bottlenecks, and how to avoid misleading benchmarks.",
)


def request_json(base: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def metrics(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {}
    names = (
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )
    for line in lines:
        for name in names:
            if line.startswith(name + "{"):
                values[name] = values.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return values


def stream_completion(base: str, body: dict) -> dict:
    request = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    first = last = None
    chunks = []
    usage = None
    finish = None
    request_id = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            request_id = event.get("id", request_id)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if text:
                    last = time.perf_counter()
                    first = last if first is None else first
                    chunks.append(text)
                finish = choice.get("finish_reason") or finish
    if usage is None or first is None or last is None:
        raise RuntimeError("Streaming completion had no text/usage; cannot score")
    completion = usage["completion_tokens"]
    text = "".join(chunks)
    return {
        "request_id": request_id,
        "usage": usage,
        "finish_reason": finish,
        "ttft_s": first - started,
        "decode_s": last - first,
        # With speculative decoding an SSE chunk can contain multiple tokens.
        # Match vLLM's conventional N-1 TPOT estimate; authoritative request
        # timing and acceptance are also saved in the server log/metrics.
        "client_decode_tok_s": (completion - 1) / (last - first) if last > first else None,
        "e2e_s": time.perf_counter() - started,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-prefix-file", type=Path)
    parser.add_argument("--draft-length", type=int, help="record the known fixed MTP width for offline round estimates")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite existing benchmark evidence")
    if args.repeats < 1 or any(length < 2 for length in args.lengths):
        raise SystemExit("Positive repeats and lengths >= 2 required")
    if args.draft_length is not None and args.draft_length < 1:
        raise SystemExit("Draft length must be positive")
    before = metrics(args.base_url)
    if any(before.get(name, 0) for name in ("vllm:num_requests_running", "vllm:num_requests_waiting")):
        raise SystemExit("Benchmark server is busy")
    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    prefix = args.prompt_prefix_file.read_text() + "\n\nTask:\n" if args.prompt_prefix_file else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for length in [32, *args.lengths]:
            for repeat in range(1 if length == 32 else args.repeats):
                before = metrics(args.base_url)
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prefix + PROMPTS[repeat % len(PROMPTS)]}],
                    "max_tokens": length,
                    "temperature": 0,
                    "seed": 42,
                    "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                result = stream_completion(args.base_url, body)
                # Stats are published by the engine periodically. Wait only
                # outside measured latency and identically for every arm.
                time.sleep(1)
                after = metrics(args.base_url)
                draft = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get(
                    "vllm:spec_decode_num_draft_tokens_total", 0
                )
                accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get(
                    "vllm:spec_decode_num_accepted_tokens_total", 0
                )
                result.update(
                    {
                        "length": length,
                        "repeat": repeat,
                        "warmup": length == 32,
                        "request": body,
                        "drafted_delta": draft,
                        "accepted_delta": accepted,
                        "acceptance": accepted / draft if draft else None,
                    }
                )
                if args.draft_length is not None:
                    result["num_speculative_tokens"] = args.draft_length
                if result["usage"]["completion_tokens"] != length:
                    raise RuntimeError(f"Requested {length} tokens but received {result['usage']}")
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(json.dumps({k: v for k, v in result.items() if k not in {"text", "request"}}), flush=True)


if __name__ == "__main__":
    main()
