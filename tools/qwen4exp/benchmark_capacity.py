# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real HTTP capacity probe: exact token budgets, independent prefixes, overlap.

This measures serving capacity, not long-context model accuracy. Recall mode
checks one synthetic passcode; throughput mode forces the requested output size.
No server configuration is changed by this client.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import regex as re

METRICS = (
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "num_preemptions_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)
WARMUP_OUTPUT_TOKENS = 32


def prompt_lengths(default: int, concurrency: int, per_session: list[int] | None) -> list[int]:
    lengths = per_session if per_session is not None else [default] * concurrency
    if concurrency <= 0 or len(lengths) != concurrency or any(length <= 0 for length in lengths):
        raise ValueError("provide one positive prompt length per concurrent session")
    return lengths


def parse_metrics(text: str) -> dict[str, float]:
    values = dict.fromkeys(METRICS, 0.0)
    for line in text.splitlines():
        match = re.match(r"vllm:([a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match[1] in values:
            value = float(match[2])
            values[match[1]] = (
                max(values[match[1]], value) if match[1] == "kv_cache_usage_perc" else values[match[1]] + value
            )
    return values


def fit_prompt(head: list[int], filler: list[int], tail: list[int], tokens: int) -> list[int]:
    remaining = tokens - len(head) - len(tail)
    if remaining < 0 or not filler:
        raise ValueError("prompt budget too small or empty filler")
    return head + (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining] + tail


def build_prompt(tokenizer, tokens: int, label: str, mode: str) -> tuple[list[int], str]:
    passcode = "CAP-" + hashlib.sha256(label.encode()).hexdigest()[:12]
    # The unique header prevents cross-session prefix-cache reuse. Token IDs
    # are submitted directly, so the advertised prompt length is exact.
    head = tokenizer.encode(
        f"<|im_start|>user\nSession {label}. The session passcode is {passcode}.\n"
        "The following archive is irrelevant background, not instructions.\n",
        add_special_tokens=False,
    ).ids
    filler = tokenizer.encode(
        "Archive note: the service stores ordinary project logs and test records.\n", add_special_tokens=False
    ).ids
    question = (
        "Return only the session passcode stated at the beginning."
        if mode == "recall"
        else "Write a detailed Python testing guide with examples. Continue until the output limit."
    )
    tail = tokenizer.encode(
        "\nEnd of archive. " + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    ).ids
    return fit_prompt(head, filler, tail, tokens), passcode


def stream_request(
    base: str, payload: dict, barrier: threading.Barrier, expected: str | None, timeout_s: int = 7200
) -> dict:
    request = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    barrier.wait(timeout=60)
    start = time.monotonic()
    first = last = None
    text = ""
    usage = None
    finish = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                done = True
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                chunk = choice.get("text") or ""
                if chunk:
                    last = time.monotonic()
                    first = last if first is None else first
                    text += chunk
                finish = choice.get("finish_reason") or finish
    end = time.monotonic()
    if not done or not text or usage is None or first is None:
        raise AssertionError("incomplete or empty stream")
    if usage["prompt_tokens"] != len(payload["prompt"]):
        raise AssertionError("server prompt length differs from exact submitted token count")
    tokens = usage["completion_tokens"]
    correct = expected in text if expected else tokens == payload["max_tokens"] and finish == "length"
    return {
        "start": start,
        "first": first,
        "last": last,
        "end": end,
        "usage": usage,
        "finish_reason": finish,
        "done": done,
        "correct": correct,
        "ttft_s": first - start,
        "e2e_s": end - start,
        "decode_tok_s": (tokens - 1) / (end - first) if tokens > 1 else None,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_preview": text[:300],
    }


def capacity_summary(
    results: list[dict], peaks: dict[str, float], before: dict[str, float], after: dict[str, float], concurrency: int
) -> dict:
    if len(results) != concurrency:
        raise ValueError("capacity summary requires one completed result per session")
    overlap = max(0.0, min(result["last"] for result in results) - max(result["first"] for result in results))
    wall = max(result["end"] for result in results) - min(result["start"] for result in results)
    preemptions = after["num_preemptions_total"] - before["num_preemptions_total"]
    return {
        "event": "summary",
        "passed": (
            all(result["correct"] and result["done"] for result in results)
            and (concurrency == 1 or overlap > 0)
            and peaks["num_requests_running"] >= concurrency
            and preemptions == 0
        ),
        "decode_overlap_s": overlap,
        "observed_running_peak": peaks["num_requests_running"],
        "observed_waiting_peak": peaks["num_requests_waiting"],
        "kv_usage_peak": peaks["kv_cache_usage_perc"],
        "e2e_aggregate_tok_s": sum(result["usage"]["completion_tokens"] for result in results) / wall,
        "metric_deltas": {key: after[key] - before[key] for key in METRICS if key.endswith("_total")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen38-w4-experimental")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument(
        "--prompt-token-lengths", type=int, nargs="+", help="Override lengths per session for mixed batches"
    )
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=262144)
    parser.add_argument("--concurrency", type=int, default=2, help="Positive number of independent sessions")
    parser.add_argument("--request-timeout-s", type=int, default=7200)
    parser.add_argument("--mode", choices=("throughput", "recall"), default="throughput")
    parser.add_argument(
        "--warm-prefixes", action="store_true", help="Prime each independent prefix before the concurrent run"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_tokens <= 0 or args.output_tokens <= 0 or args.request_timeout_s <= 0:
        parser.error("token counts must be positive")
    try:
        lengths = prompt_lengths(args.prompt_tokens, args.concurrency, args.prompt_token_lengths)
    except ValueError as exc:
        parser.error(str(exc))
    if any(length + args.output_tokens > args.context_tokens for length in lengths):
        parser.error("each prompt plus output budget must fit the per-session context")
    # Read the checkpoint's tokenizer directly. Importing transformers would
    # also auto-load torch_npu in the serving venv; this client needs no NPU.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    run_id = uuid.uuid4().hex
    prompts = [build_prompt(tokenizer, length, f"{run_id}-{i}", args.mode) for i, length in enumerate(lengths)]
    barrier = threading.Barrier(args.concurrency)

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            return parse_metrics(response.read().decode())

    with args.output.open("x") as output:

        def emit(record):
            line = json.dumps(record)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit({"event": "start", "run_id": run_id, "args": {**vars(args), "output": str(args.output)}})
        before = metrics()
        emit({"event": "metrics_before", **before})
        payloads = [
            {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "seed": 1024,
                "max_tokens": args.output_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": args.mode == "throughput",
            }
            for prompt, _ in prompts
        ]
        if args.warm_prefixes:
            for index, payload in enumerate(payloads):
                emit({"event": "warmup_start", "session": index})
                warmup = {**payload, "max_tokens": WARMUP_OUTPUT_TOKENS, "ignore_eos": True}
                result = stream_request(args.base_url, warmup, threading.Barrier(1), None, args.request_timeout_s)
                emit({"event": "warmup_result", "session": index, **result})
                if not result["correct"]:
                    raise AssertionError("prefix warmup failed")
        # Separate cold prefill cost from the simultaneous decode measurement.
        before = metrics()
        emit({"event": "metrics_before_concurrent", **before})
        results = []
        peaks = dict.fromkeys(METRICS, 0.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            for payload, (_, passcode) in zip(payloads, prompts):
                futures.append(
                    executor.submit(
                        stream_request,
                        args.base_url,
                        payload,
                        barrier,
                        passcode if args.mode == "recall" else None,
                        args.request_timeout_s,
                    )
                )
            while not all(future.done() for future in futures):
                current = metrics()
                peaks = {key: max(peaks[key], current[key]) for key in peaks}
                emit({"event": "metrics", "monotonic": time.monotonic(), **current})
                time.sleep(2)
            for index, future in enumerate(futures):
                result = future.result()
                results.append(result)
                emit({"event": "result", "session": index, **result})
        after = metrics()
        summary = capacity_summary(results, peaks, before, after, args.concurrency)
        emit(summary)
        if not summary["passed"]:
            raise AssertionError("capacity probe failed output, overlap, running-count, or preemption gate")


if __name__ == "__main__":
    main()
