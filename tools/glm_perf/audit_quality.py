"""Classify saved GLM quality results without changing the strict gate.

This audit is deliberately conservative: it only removes enclosing inline
code/quote delimiters and folds case to identify *possible* formatting-only
misses. It never changes the recorded strict pass/fail result. Incorrect
answers, parser leakage, missing finals, and invalid requests stay distinct.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.glm_perf.suite import load_quality_cases

MAX_WRAPPER_LAYERS = 2
ANSWER_WRAPPERS = (("`", "`"), ("'", "'"), ('"', '"'))


def _unwrap_answer(value: str) -> tuple[str, list[str]]:
    value = value.strip()
    wrappers: list[str] = []
    for _ in range(MAX_WRAPPER_LAYERS):
        for opener, closer in ANSWER_WRAPPERS:
            if len(value) >= 2 and value.startswith(opener) and value.endswith(closer):
                value = value[1:-1].strip()
                wrappers.append("code" if opener == "`" else "quote")
                break
        else:
            break
    return value, wrappers


def classify_row(row: dict[str, Any], expected: str) -> dict[str, Any]:
    """Add a diagnostic label; never reinterpret ``row['passed']``."""
    content = row.get("content")
    if not row.get("valid"):
        classification = "invalid_request"
        format_changes: list[str] = []
    elif row.get("raw_thinking_marker"):
        classification = "parser_leak"
        format_changes = []
    elif row.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
        classification = "missing_final"
        format_changes = []
    elif content.strip() == expected:
        classification = "exact_answer"
        format_changes = []
    else:
        unwrapped, format_changes = _unwrap_answer(content)
        if unwrapped.casefold() == expected.casefold():
            if unwrapped != expected:
                format_changes.append("case")
            classification = "possible_format_only"
        else:
            nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
            last_line = _unwrap_answer(nonempty_lines[-1])[0] if len(nonempty_lines) > 1 else ""
            classification = (
                "expected_final_with_extra_text" if last_line.casefold() == expected.casefold() else "answer_mismatch"
            )
            format_changes = []
    return {
        "case_id": row.get("case_id"),
        "strict_passed": row.get("passed") is True,
        "classification": classification,
        "format_changes": format_changes,
        "expected": expected,
        "content": content,
        "reasoning": row.get("reasoning"),
        "finish_reason": row.get("finish_reason"),
        "completion_tokens": (row.get("usage") or {}).get("completion_tokens"),
    }


def audit_file(path: Path) -> dict[str, Any]:
    quality_cases, _ = load_quality_cases()
    cases_by_id = {case.case_id: case for case in quality_cases}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            case_id = item.get("case_id")
            if case_id not in cases_by_id:
                raise ValueError(f"{path}:{line_number}: unknown quality case {case_id!r}")
            if case_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate quality case {case_id!r}")
            seen_ids.add(case_id)
            case = cases_by_id[case_id]
            expected = case.expected
            if item.get("expected") != expected:
                raise ValueError(f"{path}:{line_number}: expected answer differs from current workload")
            if "prompt" in item and item["prompt"] != case.prompt:
                raise ValueError(f"{path}:{line_number}: prompt differs from current workload")
            if not isinstance(item.get("passed"), bool):
                raise ValueError(f"{path}:{line_number}: missing strict pass/fail result")
            classified = classify_row(item, expected)
            if classified["strict_passed"] and classified["classification"] != "exact_answer":
                raise ValueError(f"{path}:{line_number}: strict pass conflicts with recorded answer")
            rows.append(classified)
    if not rows:
        raise ValueError(f"{path}: no quality rows")
    counts = Counter(row["classification"] for row in rows)
    strict_passes = sum(row["strict_passed"] for row in rows)
    return {
        "source": str(path),
        "requests": len(rows),
        "complete_suite": seen_ids == set(cases_by_id),
        "strict_passes": strict_passes,
        "strict_gate_passed": seen_ids == set(cases_by_id) and strict_passes == len(rows),
        "classifications": dict(sorted(counts.items())),
        "strict_failures": [row for row in rows if not row["strict_passed"]],
    }


def audit_files(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one quality JSONL is required")
    reports = [audit_file(path) for path in paths]
    failure_sets = [set(row["case_id"] for row in report["strict_failures"]) for report in reports]
    return {
        "strict_gate_unchanged": True,
        "format_only_is_diagnostic_not_pass": True,
        "runs": reports,
        "strict_failures_common_to_all_runs": sorted(set.intersection(*failure_sets)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="saved quality-suite JSONL files")
    parser.add_argument("--output", type=Path, help="new JSON report path; stdout if omitted")
    args = parser.parse_args()
    report = audit_files(args.inputs)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        if args.output.exists():
            parser.error("output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
