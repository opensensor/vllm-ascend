# SPDX-License-Identifier: Apache-2.0
"""Pair exact cold and repeated-prefix requests without changing the server.

Reports uncached tokens / time to first token, which includes first-token work.
It is an observed prompt-processing rate, not a kernel-only prefill rate.
"""

import argparse
import hashlib
import json
import threading
import urllib.request
import uuid
from pathlib import Path

from tools.qwen4exp.benchmark_capacity import build_prompt, stream_request


def prompt_accounting(result):
    usage = result["usage"]
    total = usage["prompt_tokens"]
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if type(total) is not int or total < 1:
        raise ValueError("invalid prompt token count")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= total):
        raise ValueError("invalid cached token count")
    duration = result["ttft_s"]
    if duration <= 0:
        raise ValueError("positive time to first token required")
    return {
        "prompt_tokens": total,
        "cached_tokens": cached,
        "uncached_tokens": None if cached is None else total - cached,
        "uncached_prompt_tok_s_including_first_token": None if cached is None else (total - cached) / duration,
    }


def pair_summary(cold, warm):
    cold_counts, warm_counts = prompt_accounting(cold), prompt_accounting(warm)
    return {
        "cold": cold_counts,
        "warm": warm_counts,
        "cold_verified": cold_counts["cached_tokens"] == 0,
        "prefix_reuse_verified": warm_counts["cached_tokens"] is not None and warm_counts["cached_tokens"] > 0,
        "recall_passed": cold["correct"] and warm["correct"],
        "same_prompt_length": cold_counts["prompt_tokens"] == warm_counts["prompt_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--label", default=None, help="Use the same fresh label for both A/B server arms")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.prompt_tokens < 128 or args.output_tokens < 2:
        parser.error("at least 128 prompt and 2 output tokens required")
    # Loading tokenizer.json directly avoids importing torch/transformers and
    # preserves the exact submitted token budget.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    prompt, expected = build_prompt(tokenizer, args.prompt_tokens, args.label or uuid.uuid4().hex, "recall")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:

        def emit(record):
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps(record), flush=True)

        emit(
            {
                "event": "request",
                "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                "prompt_tokens": len(prompt),
                "expected": expected,
                "model": model,
            }
        )
        cold = stream_request(args.base_url, payload, threading.Barrier(1), expected)
        emit({"event": "cold", **cold, **prompt_accounting(cold)})
        warm = stream_request(args.base_url, payload, threading.Barrier(1), expected)
        emit({"event": "warm", **warm, **prompt_accounting(warm)})
        summary = pair_summary(cold, warm)
        emit({"event": "summary", **summary})
    if not all(
        summary[key] for key in ("cold_verified", "prefix_reuse_verified", "recall_passed", "same_prompt_length")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
