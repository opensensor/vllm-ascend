# SPDX-License-Identifier: Apache-2.0
"""Send matched cold-prefill requests for the Qwen W4 SwiGLU experiment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from tools.qwen38_decode_study.benchmark import metrics, request_json, stream_completion
from tools.qwen38_decode_study.long_prefix import BASE_CAPACITY, CAPACITY_VARIANTS, MODULE_COUNT, TEMPLATE

LONG_CONTEXT = "The following repository excerpt is context for a programming discussion.\n\n" + "\n".join(
    TEMPLATE.format(index=index, capacity=BASE_CAPACITY + index % CAPACITY_VARIANTS) for index in range(MODULE_COUNT)
)


def prompt_for_case(case: int) -> str:
    return (
        f"Unique experiment case {case}; inspect this repository excerpt.\n\n"
        + LONG_CONTEXT
        + "\n\nSummarize how a caller should shut down the worker queue safely."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("swiglu_fused", "builtin_fp16"), required=True)
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()
    if args.cases <= 0 or args.output.exists():
        parser.error("--cases must be positive and --output must be a new path")
    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    with args.output.open("x") as output:
        for case in range(args.cases):
            prompt = prompt_for_case(case)
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            before = metrics(args.base_url)
            result = stream_completion(args.base_url, body)
            time.sleep(1)
            after = metrics(args.base_url)
            drafted = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get(
                "vllm:spec_decode_num_draft_tokens_total", 0
            )
            accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get(
                "vllm:spec_decode_num_accepted_tokens_total", 0
            )
            result.update(
                {
                    "arm": args.arm,
                    "case": case,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_chars": len(prompt),
                    "drafted_delta": drafted,
                    "accepted_delta": accepted,
                    "acceptance": accepted / drafted if drafted else None,
                }
            )
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(json.dumps({key: value for key, value in result.items() if key != "text"}), flush=True)


if __name__ == "__main__":
    main()
