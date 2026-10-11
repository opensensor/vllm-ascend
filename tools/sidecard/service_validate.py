# SPDX-License-Identifier: Apache-2.0
"""Exercise actual generation and prove CUDA routed execution in every GLM layer."""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "glm-sidecard"


def request(base_url, endpoint, body=None, timeout=300):
    raw = None if body is None else json.dumps(body).encode()
    query = urllib.request.Request(
        base_url.rstrip("/") + endpoint, data=raw, headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    with urllib.request.urlopen(query, timeout=timeout) as response:
        result = json.load(response)
        status = response.status
    if status != 200:
        raise AssertionError(f"service returned HTTP {status}")
    return {"elapsed_seconds": time.monotonic() - start, "status": status, "response": result}


def completion_text(result):
    response = result["response"]
    text = response["choices"][0]["text"]
    if not text.strip() or response["usage"]["completion_tokens"] < 1:
        raise AssertionError("completion request returned no generated text")
    return text


def check_metrics(value):
    layers = value["layers"]
    if set(layers) != {str(layer) for layer in range(3, 45)}:
        raise AssertionError("service did not install every main-model routed layer")
    if any(row["real_calls"] < 1 or row["real_cuda_routes"] < 1 or row["shadow_checks"] < 3 for row in layers.values()):
        raise AssertionError("service lacks real CUDA routes or real-input shadow validation in a routed layer")
    return {
        "validated_layers": len(layers),
        "real_cuda_routes": sum(row["real_cuda_routes"] for row in layers.values()),
        "real_input_shadow_checks": sum(row["shadow_checks"] for row in layers.values()),
        "shadow_max_absolute_error": max(row["shadow_max_absolute_error"] for row in layers.values()),
        "shadow_max_nrmse": max(row["shadow_max_nrmse"] for row in layers.values()),
        "cuda_worker_pid": value["cuda_worker_pid"],
        "profile": value["profile"],
        "cuda_attention": value["cuda_attention"],
        "retains_npu_shadow_banks": value["retains_npu_shadow_banks"],
    }


def run(base_url, metrics_path, output, baseline=None, *, require_repeat=True, prefill_budget=128):
    if type(prefill_budget) is not int or prefill_budget < 1:
        raise ValueError("positive prefill budget required")
    report = {
        "schema_version": 1,
        "base_url": base_url,
        "model": MODEL,
        "passed": False,
        "operational_passed": False,
        "cases": [],
    }
    before = json.loads(Path(metrics_path).read_bytes())["layers"]
    try:
        models = request(base_url, "/v1/models")
        if MODEL not in {row["id"] for row in models["response"]["data"]}:
            raise AssertionError("sidecard model name is absent from /v1/models")
        report["models"] = models
        body = dict(model=MODEL, prompt="The capital of France is", max_tokens=12, temperature=0, logprobs=1)
        first = request(base_url, "/v1/completions", body)
        text = completion_text(first)
        report["cases"].append({"name": "completion", **first})
        if baseline is not None:
            previous = json.loads(Path(baseline).read_bytes())
            report["baseline_text_equal"] = text == previous["choices"][0]["text"]
        chat = request(
            base_url,
            "/v1/chat/completions",
            dict(
                model=MODEL,
                messages=[dict(role="user", content="What is 2 + 2? Answer briefly.")],
                max_tokens=32,
                temperature=0,
            ),
        )
        message = chat["response"]["choices"][0]["message"]
        if not (message.get("content") or message.get("reasoning")):
            raise AssertionError("chat completion returned no generated text")
        report["cases"].append({"name": "chat", **chat})
        repeat = request(base_url, "/v1/completions", body)
        report["deterministic_repeat_equal"] = completion_text(repeat) == text
        report["cases"].append({"name": "repeat", **repeat})
        long_body = dict(
            model=MODEL,
            prompt="Hello world. " * max(80, prefill_budget // 3 + 16) + "Continue: ",
            max_tokens=4,
            temperature=0,
        )
        long_result = request(base_url, "/v1/completions", long_body)
        completion_text(long_result)
        if long_result["response"]["usage"]["prompt_tokens"] <= prefill_budget:
            raise AssertionError("long prompt did not exercise chunked prefill")
        report["cases"].append({"name": "chunked-prefill", **long_result})
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    request, base_url, "/v1/completions", dict(model=MODEL, prompt=prompt, max_tokens=6, temperature=0)
                )
                for prompt in ("The opposite of hot is", "One plus one equals")
            ]
            for index, future in enumerate(futures):
                result = future.result()
                completion_text(result)
                report["cases"].append({"name": f"concurrent-{index}", **result})
        query = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(
                dict(model=MODEL, prompt="Counting: one, two,", max_tokens=4, temperature=0, stream=True)
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        events, done = [], False
        with urllib.request.urlopen(query, timeout=300) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    done = True
                else:
                    events.append(json.loads(payload))
        if not done or not any(event.get("choices", [{}])[0].get("text") for event in events):
            raise AssertionError("stream did not emit generated text and a DONE event")
        report["cases"].append({"name": "stream", "done": done, "events": events})
        after = json.loads(Path(metrics_path).read_bytes())
        report["model_boundary"] = check_metrics(after)
        deltas = {
            layer: row["real_cuda_routes"] - before[layer]["real_cuda_routes"] for layer, row in after["layers"].items()
        }
        if any(count <= 0 for count in deltas.values()):
            raise AssertionError("an actual API validation request did not reach CUDA in every routed layer")
        report["actual_api_cuda_route_deltas"] = deltas
        report["operational_passed"] = True
        report["passed"] = report["deterministic_repeat_equal"]
        if require_repeat and not report["passed"]:
            raise AssertionError("operational requests passed, but greedy repeated completion changed")
    finally:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prefill-budget", type=int, default=128)
    parser.add_argument(
        "--operational-only",
        action="store_true",
        help="Complete API checks and report repeat variation without claiming a full validation pass",
    )
    args = parser.parse_args()
    report = run(
        args.base_url,
        args.metrics_path,
        args.output,
        args.baseline,
        require_repeat=not args.operational_only,
        prefill_budget=args.prefill_budget,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "operational_passed": report["operational_passed"],
                "cases": len(report["cases"]),
                "model_boundary": report["model_boundary"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
