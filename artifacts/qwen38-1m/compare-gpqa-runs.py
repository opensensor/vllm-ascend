#!/usr/bin/env python3
"""Compare two AISBench GPQA Diamond result directories."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def load_run(root: Path, cap: int) -> dict[int, dict]:
    result_path = find_one(root, "**/results/*/GPQA_diamond.json")
    prediction_path = find_one(root, "**/predictions/*/GPQA_diamond.jsonl")
    details = json.loads(result_path.read_text(encoding="utf-8"))["details"]
    with prediction_path.open(encoding="utf-8") as prediction_file:
        predictions = {
            int(row["id"]): row
            for row in (
                json.loads(line) for line in prediction_file if line.strip()
            )
        }

    rows = {}
    for raw_id, detail in details.items():
        item_id = int(raw_id)
        prediction = predictions[item_id]
        extracted = detail["predictions"][0]
        rows[item_id] = {
            "correct": bool(detail["correct"][0]),
            "extracted": extracted,
            "gold": detail["references"][0],
            "output_tokens": int(prediction.get("output_tokens", 0)),
            "at_cap": int(prediction.get("output_tokens", 0)) >= cap,
            "prompt": prediction["origin_prompt"][0]["prompt"],
        }
    return rows


def summarize(rows: dict[int, dict]) -> dict:
    values = list(rows.values())
    answered = [row for row in values if row["extracted"] is not None]
    output_tokens = [row["output_tokens"] for row in values]
    correct = sum(row["correct"] for row in values)
    answered_correct = sum(row["correct"] for row in answered)
    return {
        "questions": len(values),
        "correct": correct,
        "raw_accuracy": 100 * correct / len(values),
        "answered": len(answered),
        "answered_correct": answered_correct,
        "answered_accuracy": 100 * answered_correct / len(answered),
        "at_cap": sum(row["at_cap"] for row in values),
        "cap_without_answer": sum(
            row["at_cap"] and row["extracted"] is None for row in values
        ),
        "output_tokens": sum(output_tokens),
        "median_output_tokens": statistics.median(output_tokens),
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def question_from_prompt(prompt: str) -> str:
    question = prompt.split("Think step by step before answering.\n\n", 1)[-1]
    return question.rsplit("\n\nA)", 1)[0]


def load_domains(path: Path) -> dict[str, str]:
    rows = csv.DictReader(path.open(encoding="utf-8", newline=""))
    domains = {}
    for row in rows:
        for key in ("Question", "Extra Revised Question", "Pre-Revision Question"):
            if row.get(key):
                domains[normalize(row[key])] = row["High-level domain"]
    return domains


def domain_summary(rows: dict[int, dict], domains: dict[str, str]) -> dict:
    grouped = defaultdict(list)
    for row in rows.values():
        question = normalize(question_from_prompt(row["prompt"]))
        domain = domains.get(question)
        if domain is None:
            match = difflib.get_close_matches(question, domains, n=1, cutoff=0.9)
            domain = domains[match[0]] if match else "Unmatched"
        grouped[domain].append(row)
    return {domain: summarize(dict(enumerate(items))) for domain, items in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--cap", type=int, default=8192)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load_run(args.baseline, args.cap)
    candidate = load_run(args.candidate, args.cap)
    if baseline.keys() != candidate.keys():
        raise SystemExit("Run item IDs differ; refusing a non-paired comparison")

    base_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    incorrect_to_correct = sorted(
        item_id
        for item_id in baseline
        if not baseline[item_id]["correct"] and candidate[item_id]["correct"]
    )
    correct_to_incorrect = sorted(
        item_id
        for item_id in baseline
        if baseline[item_id]["correct"] and not candidate[item_id]["correct"]
    )

    lines = [
        "# GPQA Diamond paired comparison",
        "",
        "| Metric | RTX llama.cpp baseline | Ascend candidate |",
        "| --- | ---: | ---: |",
    ]
    metrics = (
        ("Raw accuracy", "raw_accuracy", "%"),
        ("Correct", "correct", ""),
        ("Extracted choices", "answered", ""),
        ("Answered-only accuracy", "answered_accuracy", "%"),
        ("Outputs at token cap", "at_cap", ""),
        ("Cap without answer", "cap_without_answer", ""),
        ("Total output tokens", "output_tokens", ""),
        ("Median output tokens", "median_output_tokens", ""),
    )
    for label, key, suffix in metrics:
        left = base_summary[key]
        right = candidate_summary[key]
        if suffix:
            left = f"{left:.2f}{suffix}"
            right = f"{right:.2f}{suffix}"
        lines.append(f"| {label} | {left} | {right} |")

    lines.extend(
        [
            "",
            "## Item-level flips",
            "",
            f"- Incorrect → correct: {len(incorrect_to_correct)} — {incorrect_to_correct}",
            f"- Correct → incorrect: {len(correct_to_incorrect)} — {correct_to_incorrect}",
        ]
    )

    if args.metadata_csv:
        domains = load_domains(args.metadata_csv)
        base_domains = domain_summary(baseline, domains)
        candidate_domains = domain_summary(candidate, domains)
        lines.extend(
            [
                "",
                "## Domain results",
                "",
                "| Domain | Baseline raw | Ascend raw | Baseline answered | Ascend answered |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for domain in sorted(set(base_domains) | set(candidate_domains)):
            left = base_domains[domain]
            right = candidate_domains[domain]
            lines.append(
                f"| {domain} | {left['raw_accuracy']:.1f}% | "
                f"{right['raw_accuracy']:.1f}% | {left['answered_accuracy']:.1f}% | "
                f"{right['answered_accuracy']:.1f}% |"
            )

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
