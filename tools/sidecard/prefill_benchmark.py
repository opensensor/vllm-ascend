# SPDX-License-Identifier: Apache-2.0
"""Measure completed cold and repeated-prefix prefill through the actual API."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from .service_validate import MODEL, request


def exact_prompt(tokens, length):
    if not tokens or type(length) is not int or length < 1:
        raise ValueError("positive prompt length and tokenizer output required")
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def stream_measure(base_url, prompt, *, timeout=600):
    query = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(
            dict(
                model=MODEL,
                prompt=prompt,
                max_tokens=4,
                temperature=0,
                stream=True,
                stream_options=dict(include_usage=True),
            )
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_text = None
    text = ""
    usage = None
    done = False
    with urllib.request.urlopen(query, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                done = True
                continue
            event = json.loads(payload)
            for choice in event.get("choices", []):
                fragment = choice.get("text", "")
                if fragment and first_text is None:
                    first_text = time.monotonic() - started
                text += fragment
            if event.get("usage"):
                usage = event["usage"]
    if not done or first_text is None or usage is None:
        raise AssertionError("stream did not complete with generated text and token usage")
    if usage["prompt_tokens"] != len(prompt):
        raise AssertionError("server prompt usage differs from requested token count")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return dict(
        prompt_tokens=len(prompt),
        cached_tokens=cached,
        usage=usage,
        text=text,
        time_to_first_text_seconds=first_text,
        elapsed_seconds=time.monotonic() - started,
        uncached_prompt_tokens_per_second=(len(prompt) - cached) / first_text,
        generation_completed=True,
    )


def run(base_url, lengths, output, *, timeout=600):
    report = dict(passed=False, model=MODEL, cases=[], long_context_quality_validated=False)
    try:
        tokens = request(
            base_url,
            "/tokenize",
            dict(
                model=MODEL,
                prompt="Review this Python module. Describe its functions and identify errors.\n",
                add_special_tokens=False,
            ),
        )["response"]["tokens"]
        for length in lengths:
            prompt = exact_prompt(tokens, length)
            # The first case may reuse a previous shorter case; record actual cached usage.
            for label in ("first", "repeat"):
                result = stream_measure(base_url, prompt, timeout=timeout)
                report["cases"].append(dict(name=f"{length}-{label}", **result))
                Path(output).write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024])
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.lengths, args.output, timeout=args.timeout)), flush=True)


if __name__ == "__main__":
    main()
