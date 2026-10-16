# SPDX-License-Identifier: Apache-2.0
"""Repeatable serial and six-request serving measurements for TP6 candidates."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.qwen38_decode_study.benchmark import PROMPTS, stream_completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--prefix-file", type=Path)
    parser.add_argument("--skip-concurrent", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.tokens < 2:
        parser.error("use a new output file and at least two tokens")
    prefix = "" if args.prefix_file is None else args.prefix_file.read_text() + "\n"

    def completion(index, tokens):
        result = stream_completion(
            args.base_url,
            dict(
                model="qwen38-flash-next",
                messages=[dict(role="user", content=prefix + PROMPTS[index % len(PROMPTS)])],
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
            raise RuntimeError("completion token budget did not match")
        return result

    with args.output.open("x") as output:
        completion(0, 32)
        for index in range(len(PROMPTS)):
            result = completion(index, args.tokens)
            output.write(json.dumps(dict(kind="serial", index=index, result=result)) + "\n")
            output.flush()
            print(index, result["client_decode_tok_s"], result["text_sha256"], flush=True)
        if not args.skip_concurrent:
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=6) as pool:
                rows = list(pool.map(lambda index: completion(index, args.tokens), range(6)))
            elapsed = time.perf_counter() - start
            throughput = 6 * args.tokens / elapsed
            output.write(
                json.dumps(dict(kind="parallel", elapsed_s=elapsed, aggregate_tok_s=throughput, results=rows)) + "\n"
            )
            print("C6 aggregate end-to-end", throughput, flush=True)


if __name__ == "__main__":
    main()
