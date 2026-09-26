# SPDX-License-Identifier: Apache-2.0
"""Repeatable long-prefill request for real-weight 310P QSA service checks."""

import argparse
import json
import time
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--repeats", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--question", default="")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")

    payload = {
        "model": "qwen38-flash-next-w8a8",
        "prompt": "alpha beta gamma delta " * args.repeats + args.question,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "logprobs": 5,
    }
    request = Request(args.url, json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=900) as response:
        result = json.load(response)
    choice = result["choices"][0]
    first_logprobs = choice.get("logprobs") or {}
    top_logprobs = first_logprobs.get("top_logprobs") or []
    print(
        json.dumps(
            {
                "seconds": round(time.perf_counter() - started, 3),
                "usage": result["usage"],
                "finish_reason": choice["finish_reason"],
                "text": choice["text"][-512:],
                "first_token_logprobs": top_logprobs[0] if top_logprobs else None,
            }
        )
    )


if __name__ == "__main__":
    main()
