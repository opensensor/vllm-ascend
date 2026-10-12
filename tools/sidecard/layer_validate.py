# SPDX-License-Identifier: Apache-2.0
"""Run completed API, repeatability, arithmetic, and context-capacity gates."""

import argparse
import json
import urllib.error
from pathlib import Path

import yaml

from .decode_benchmark import run as decode_run
from .prefill_benchmark import exact_prompt, stream_measure
from .service_validate import request

ARITHMETIC = (("7 * 8", "56"), ("19 + 23", "42"), ("144 / 12", "12"), ("9 * 9", "81"), ("100 - 37", "63"))


def evaluation_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("model_type") != "sidecard-layer-runtime" or config.get("num_fewshot") != 0:
        raise ValueError("this gate requires a zero-shot sidecard-layer-runtime configuration")
    if config.get("tasks") != [
        {"name": "sidecard_arithmetic_smoke", "metrics": [{"name": "exact_match", "value": 1.0}]}
    ]:
        raise ValueError("this gate supports only the documented five-prompt arithmetic smoke")
    return config


def run(base_url, output, *, lengths=(), output_tokens=64):
    output = Path(output)
    report = {
        "passed": False,
        "dummy_weights": False,
        "cases": [],
        "long_context_quality_validated": False,
        "accuracy_scope": "five elementary arithmetic prompts; no general benchmark claim",
    }

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["health"] = request(base_url, "/health")["response"]
        models = request(base_url, "/v1/models")["response"]
        assert models["data"][0]["id"] == "glm-sidecard"
        correct = 0
        for expression, answer in ARITHMETIC:
            response = request(
                base_url,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": f"Calculate {expression}. Reply with only the number."}],
                    "reasoning_effort": "low",
                    "max_tokens": 64,
                    "temperature": 0,
                },
            )["response"]
            actual = response["choices"][0]["message"]["content"].strip()
            passed = actual == answer
            correct += passed
            report["cases"].append(
                dict(name="arithmetic", expression=expression, expected=answer, actual=actual, passed=passed)
            )
            save()
        report["arithmetic_accuracy"] = correct / len(ARITHMETIC)
        answers = []
        for index in range(3):
            response = request(
                base_url,
                "/v1/completions",
                {
                    "prompt": 'def factorial(n):\n    """Compute factorial recursively."""\n',
                    "max_tokens": output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            )["response"]
            assert response["usage"]["completion_tokens"] == output_tokens
            answers.append(response["choices"][0]["text"])
            report["cases"].append(dict(name=f"repeat-{index}", response=response))
            save()
        report["repeat_text_equal"] = all(answer == answers[0] for answer in answers)
        assert report["repeat_text_equal"]
        # The matching historical reference uses this exact 2048/64 fixture.
        report["decode2048"] = decode_run(
            base_url, output.with_name(output.stem + "-decode2048.json"), output_tokens=output_tokens
        )
        tokens = request(base_url, "/tokenize", {"prompt": "Inspect this module and explain its control flow. "})[
            "response"
        ]["tokens"]
        for length in lengths:
            result = stream_measure(base_url, exact_prompt(tokens, length), timeout=7200, max_tokens=8, ignore_eos=True)
            assert result["usage"]["completion_tokens"] == 8
            report["cases"].append(dict(name=f"capacity-{length}", **result))
            save()
        for body in (
            {"prompt": [1], "max_tokens": report["health"]["max_model_len"]},
            {"prompt": "hi", "tools": [{}]},
            {"prompt": "hi", "temperature": 1},
        ):
            try:
                request(base_url, "/v1/completions", body)
            except urllib.error.HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("invalid request was admitted")
        report["admission_rejections_passed"] = True
        report["passed"] = correct == len(ARITHMETIC)
        if not report["passed"]:
            raise AssertionError("arithmetic smoke failed")
        return report
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="*", type=int, default=[])
    parser.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args()
    config = evaluation_config(args.evaluation_config) if args.evaluation_config else {}
    base_url = args.base_url or config.get("serve", {}).get("base_url", "http://127.0.0.1:8000")
    result = run(base_url, args.output, lengths=args.lengths, output_tokens=args.output_tokens)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
