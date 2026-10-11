# SPDX-License-Identifier: Apache-2.0
"""Measure a completed fixed-length decode separately from prompt processing."""

import argparse
import json
from pathlib import Path

from .prefill_benchmark import exact_prompt, stream_measure
from .service_validate import MODEL, request


def run(base_url, output, *, prompt_tokens=2048, output_tokens=64):
    tokens = request(
        base_url,
        "/tokenize",
        dict(
            model=MODEL,
            prompt="Inspect this module for profiler test 20261011 and explain the control flow. ",
            add_special_tokens=False,
        ),
    )["response"]["tokens"]
    result = stream_measure(base_url, exact_prompt(tokens, prompt_tokens), max_tokens=output_tokens, ignore_eos=True)
    report = dict(passed=result["usage"]["completion_tokens"] == output_tokens, **result)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError("decode did not complete the requested output budget")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(run(args.base_url, args.output, prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens)),
        flush=True,
    )


if __name__ == "__main__":
    main()
