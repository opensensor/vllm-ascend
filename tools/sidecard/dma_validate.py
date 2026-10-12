# SPDX-License-Identifier: Apache-2.0
"""Completed direct-DMA API gates for queue isolation and cancellation."""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .service_validate import request

OUTPUT_TOKENS = 8
CANCEL_RECOVERY_SECONDS = 15


def stream_request(base_url, body):
    query = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(query, timeout=300)


def run(base_url, output):
    """Compare queued completions with isolated controls; cancel at both boundaries."""
    report = {"passed": False, "cases": [], "near_capacity_prefill_completed": False}

    def save():
        Path(output).write_text(json.dumps(report, indent=2) + "\n")

    def complete(body):
        return request(base_url, "/v1/completions", body)

    try:
        health = request(base_url, "/health")["response"]
        if health.get("expert_transport") != "direct-dma":
            raise ValueError("API is not using direct DMA")
        report["health"] = health
        bodies = [
            dict(prompt=prompt, max_tokens=OUTPUT_TOKENS, temperature=0, ignore_eos=True)
            for prompt in ("The opposite of hot is", "One plus one equals")
        ]
        controls = [complete(body) for body in bodies]
        with ThreadPoolExecutor(max_workers=2) as executor:
            queued = list(executor.map(complete, bodies))
        equal = all(
            a["response"]["choices"] == b["response"]["choices"] and a["response"]["usage"] == b["response"]["usage"]
            for a, b in zip(controls, queued, strict=True)
        )
        report["cases"].append(dict(name="concurrent_isolation", passed=equal, controls=controls, queued=queued))
        if not equal:
            raise AssertionError("queued requests changed their isolated completions")
        save()
        body = dict(bodies[0], stream=True, stream_options={"include_usage": True})
        text, usage, done = "", None, False
        with stream_request(base_url, body) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                value = raw[6:].strip()
                if value == b"[DONE]":
                    done = True
                else:
                    frame = json.loads(value)
                    if frame["choices"]:
                        text += frame["choices"][0]["text"]
                    if frame.get("usage"):
                        usage = frame["usage"]
        stream_equal = done and text == controls[0]["response"]["choices"][0]["text"]
        stream_equal = stream_equal and usage == controls[0]["response"]["usage"]
        report["cases"].append(dict(name="stream_equivalence", passed=stream_equal, text=text, usage=usage, done=done))
        if not stream_equal:
            raise AssertionError("streamed completion differs from isolated control")
        save()
        token = request(base_url, "/tokenize", {"prompt": "Hello", "add_special_tokens": False})["response"]["tokens"][
            0
        ]
        for name, cancellation in (
            ("cancel_before_prefill", dict(prompt=[token] * (health["max_model_len"] - 1), max_tokens=1, stream=True)),
            ("cancel_during_decode", dict(prompt=bodies[0]["prompt"], max_tokens=128, stream=True, ignore_eos=True)),
        ):
            with stream_request(base_url, cancellation) as response:
                if name == "cancel_during_decode":
                    for raw in response:
                        if raw.startswith(b"data: ") and json.loads(raw[6:])["choices"][0].get("text"):
                            break
            start = time.monotonic()
            followup = complete(bodies[0])
            elapsed = time.monotonic() - start
            passed = followup["response"]["choices"] == controls[0]["response"]["choices"]
            passed = passed and elapsed < CANCEL_RECOVERY_SECONDS
            report["cases"].append(
                dict(
                    name=name,
                    passed=passed,
                    followup=followup,
                    recovery_seconds=elapsed,
                    cancelled_prompt_tokens=health["max_model_len"] - 1 if name == "cancel_before_prefill" else None,
                )
            )
            save()
            if not passed:
                raise AssertionError("cancelled request blocked or corrupted the next request")
        budget = request(
            base_url,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "What is 2 + 2? Reply with only the number."}],
                "reasoning_effort": "low",
                "max_tokens": 32000,
                "temperature": 0,
            },
        )
        if budget["response"]["choices"][0]["message"]["content"].strip() != "4":
            raise AssertionError("large allowed output budget changed arithmetic generation")
        report["cases"].append(dict(name="kilo_output_budget", passed=True, result=budget))
        try:
            complete(dict(prompt=[token], max_tokens=health["max_model_len"]))
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
            report["cases"].append(dict(name="over_context_rejected", passed=True, status=error.code))
        else:
            raise AssertionError("over-context request was admitted")
        report["passed"] = True
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base_url, args.output)
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"])}), flush=True)


if __name__ == "__main__":
    main()
