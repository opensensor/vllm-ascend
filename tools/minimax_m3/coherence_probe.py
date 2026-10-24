# SPDX-License-Identifier: Apache-2.0
"""Small exact-answer HTTP smoke checks; not a broad language benchmark."""

import argparse
import base64
import json
from pathlib import Path

from tools.minimax_m3.http_probe import stream_request


def answer_matches(text, expected):
    if isinstance(expected, dict):
        try:
            return json.loads(text) == expected
        except json.JSONDecodeError:
            return False
    return text.strip().casefold().rstrip(".!") == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="minimax-m31-flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="Solid red PNG for the color check")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--extended", action="store_true", help="Add JSON, history, multi-step and sparse retrieval checks"
    )
    parser.add_argument("--case", action="append", help="Run only a named case; may be repeated")
    parser.add_argument("--records", type=int, default=180, help="Record count for the sparse retrieval case")
    args = parser.parse_args()
    if args.records < 38:
        parser.error("records must be at least 38")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("greeting", "Say hello in English, in one short sentence.", "hello"),
        ("france", "What is the capital of France? Answer with only the city name.", "paris"),
        ("addition", "What is 17 + 25? Answer with only the number.", "42"),
        ("copy", "Reply with exactly this word: cobalt", "cobalt"),
    ]
    if args.image:
        url = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
        cases.append(
            (
                "image",
                [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "Name the dominant color in this image. Use only one color word."},
                ],
                "red",
            )
        )
    if args.extended:
        records = [
            f"Record {i}: Routine inventory records mention steel bolts, paper labels, and wooden crates."
            for i in range(args.records)
        ]
        records[37] = "Record 37: The verification phrase is amber cloud."
        cases.extend(
            [
                (
                    "json",
                    'Return only valid JSON: an object with key "colors" and value ["red", "blue"].',
                    {"colors": ["red", "blue"]},
                ),
                (
                    "multi_step",
                    "A box has 3 red marbles and 2 blue marbles. I add 4 blue marbles and remove 1 red marble. "
                    "How many marbles remain? Reply with only the number.",
                    "8",
                ),
                (
                    "multi_turn",
                    [
                        {"role": "user", "content": "Remember the code cedar-47."},
                        {"role": "assistant", "content": "I will remember cedar-47."},
                        {"role": "user", "content": "What code did I ask you to remember? Reply with only the code."},
                    ],
                    "cedar-47",
                ),
                (
                    "sparse_retrieval",
                    "Read these records.\n"
                    + "\n".join(records)
                    + "\nWhat verification phrase appears in record 37? Reply with only the phrase.",
                    "amber cloud",
                ),
            ]
        )
    results = []
    for name, content, expected in cases:
        if args.case and name not in args.case:
            continue
        messages = (
            content if isinstance(content, list) and content[0].get("role") else [{"role": "user", "content": content}]
        )
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        result = stream_request(args.base_url, "/v1/chat/completions", payload, 1200)
        # Exact short answers deliberately reject leaked reasoning markup.
        result.update(name=name, expected=expected, passed=answer_matches(result["text"], expected), request=payload)
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        summary = {k: v for k, v in result.items() if k not in ("chunks", "request")}
        results.append(summary)
        print(json.dumps(summary), flush=True)
    if not results:
        parser.error("No matching cases selected")
    report = {
        "scope": "Small deterministic coherence/format smoke, not lm-eval or long-context qualification",
        "cases": results,
        "sample_count": len(results),
        "passed_fraction": sum(r["passed"] for r in results) / len(results),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
