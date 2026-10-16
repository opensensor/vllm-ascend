# SPDX-License-Identifier: Apache-2.0
"""Exercise live selection shadow checks and early draft at all TP6 batch sizes.

First enable --prefix --verify with apply_tp6_overlap.py. This client records
request completion; the worker controller owns exact selection comparisons.
Keep the controller's subsequent all-rank receipt as the numerical gate.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.qwen38_decode_study.benchmark import stream_completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous verification evidence")

    def request(index, tokens=16, long=False):
        prefix = "alpha beta gamma delta " * 700 + "\n" if long else ""
        text = prefix + f"Case {index}: explain why a condition variable needs a predicate loop."
        result = stream_completion(
            args.base_url,
            dict(
                model="qwen38-flash-next",
                messages=[dict(role="user", content=text)],
                temperature=0,
                seed=42,
                max_tokens=tokens,
                ignore_eos=True,
                chat_template_kwargs=dict(enable_thinking=False),
                stream=True,
                stream_options=dict(include_usage=True),
            ),
        )
        if result["usage"]["completion_tokens"] != tokens:
            raise RuntimeError("incomplete verification request")
        return result

    with args.output.open("x") as output:
        for count in (1, 2, 4, 6):
            with ThreadPoolExecutor(max_workers=count) as pool:
                rows = list(pool.map(request, range(count)))
            output.write(json.dumps(dict(requests=count, results=rows)) + "\n")
            output.flush()
        result = request(99, 32, True)
        output.write(json.dumps(dict(long=True, result=result)) + "\n")
        output.write(json.dumps(dict(passed=True)) + "\n")
    print("Live verification requests completed; check all six worker receipts.", flush=True)


if __name__ == "__main__":
    main()
