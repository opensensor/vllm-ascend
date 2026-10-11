# SPDX-License-Identifier: Apache-2.0
"""Check advertised context, request admission and the Kilo output budget.

The near-capacity stream is deliberately disconnected after HTTP admission.
It does not complete prefill and cannot establish long-context model quality.
Use service_validate separately for actual CUDA route and generation checks.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .service_validate import MODEL, request

KILO_OUTPUT_TOKENS = 32000


def model_limits(models):
    matching = [row for row in models["data"] if row["id"] == MODEL]
    if len(matching) != 1:
        raise ValueError("expected one sidecard model")
    context = matching[0]["max_model_len"]
    if type(context) is not int or context <= KILO_OUTPUT_TOKENS:
        raise ValueError("sidecard context cannot accommodate Kilo's output budget and input")
    return {"context": context, "input": context - KILO_OUTPUT_TOKENS, "output": KILO_OUTPUT_TOKENS}


def run(base_url, output, *, expected_context=None):
    report = {"passed": False, "full_context_prefill_completed": False, "long_context_quality_validated": False}
    try:
        models = request(base_url, "/v1/models")
        limits = model_limits(models["response"])
        if expected_context is not None and limits["context"] != expected_context:
            raise ValueError("service context differs from expected admission")
        report["limits"] = limits
        chat = request(
            base_url,
            "/v1/chat/completions",
            dict(
                model=MODEL,
                messages=[dict(role="user", content="What is 2 + 2? Reply with only the number.")],
                reasoning_effort="low",
                max_tokens=limits["output"],
                temperature=0,
            ),
        )
        message = chat["response"]["choices"][0]["message"]
        if not message.get("content") or chat["response"]["choices"][0]["finish_reason"] != "stop":
            raise AssertionError("large output-budget chat did not complete")
        report["kilo_output_budget_generation"] = chat
        tokenized = request(base_url, "/tokenize", dict(model=MODEL, prompt="Hello", add_special_tokens=False))
        tokens = tokenized["response"]["tokens"]
        if not tokens:
            raise AssertionError("tokenizer returned no input token")
        prompt_length = limits["context"] - 1
        body = dict(model=MODEL, prompt=[tokens[0]] * prompt_length, max_tokens=1, temperature=0, stream=True)
        query = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(query, timeout=120) as response:
            if response.status != 200:
                raise AssertionError("near-capacity request was not admitted")
            report["near_capacity_admission"] = {
                "status": response.status,
                "prompt_tokens": prompt_length,
                "max_tokens": 1,
                "elapsed_seconds": time.monotonic() - started,
                "disconnected_after_headers": True,
                "generation_completed": False,
            }
        # Do not fill an over-limit prompt or silently reduce the client's budget.
        rejected = dict(model=MODEL, prompt=[tokens[0]], max_tokens=limits["context"] + 1)
        query = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(rejected).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(query, timeout=120):
                raise AssertionError("over-context output budget was accepted")
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
            report["over_context_rejected"] = {"status": error.code, "response": json.loads(error.read())}
        report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-context", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.output, expected_context=args.expected_context)), flush=True)


if __name__ == "__main__":
    main()
