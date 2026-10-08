"""CPU-only checks for saved GLM quality-result triage."""

from __future__ import annotations

import json

import pytest

from tools.glm_perf.audit_quality import audit_files, classify_row


def _row(case_id: str, expected: str, content: str, *, passed: bool = False) -> dict:
    return {
        "case_id": case_id,
        "expected": expected,
        "content": content,
        "reasoning": "",
        "finish_reason": "stop",
        "usage": {"completion_tokens": 4},
        "valid": True,
        "passed": passed,
        "raw_thinking_marker": False,
    }


def test_format_labels_never_change_strict_result() -> None:
    red = classify_row(_row("instr_first", "red", "Red"), "red")
    quoted = classify_row(_row("code_slice", "lan", "`'lan'`"), "lan")
    wrong = classify_row(_row("instr_reverse", "pmal", "pial"), "pmal")
    assert red["classification"] == "possible_format_only"
    assert red["format_changes"] == ["case"]
    assert quoted["classification"] == "possible_format_only"
    assert quoted["format_changes"] == ["code", "quote"]
    assert wrong["classification"] == "answer_mismatch"
    assert not any(item["strict_passed"] for item in (red, quoted, wrong))


def test_expected_final_with_extra_text_is_still_strict_failure() -> None:
    row = _row("instr_reverse", "pmal", "maip... wait, let me check\npmal")
    classified = classify_row(row, "pmal")
    assert classified["classification"] == "expected_final_with_extra_text"
    assert not classified["strict_passed"]


def test_parser_leak_and_missing_final_are_not_format_only() -> None:
    leaked = _row("instr_first", "red", "</think>red")
    leaked["raw_thinking_marker"] = True
    assert classify_row(leaked, "red")["classification"] == "parser_leak"
    truncated = _row("instr_first", "red", "Red")
    truncated["finish_reason"] = "length"
    assert classify_row(truncated, "red")["classification"] == "missing_final"
    invalid = _row("instr_first", "red", "Red")
    invalid["valid"] = False
    assert classify_row(invalid, "red")["classification"] == "invalid_request"


def test_audit_compares_runs_without_relaxing_gate(tmp_path) -> None:
    first = tmp_path / "eager.jsonl"
    second = tmp_path / "graph.jsonl"
    first_rows = [_row("instr_first", "red", "Red"), _row("instr_reverse", "pmal", "pial")]
    second_rows = [_row("instr_first", "red", "red", passed=True), _row("instr_reverse", "pmal", "pial")]
    first.write_text("\n".join(json.dumps(item) for item in first_rows))
    second.write_text("\n".join(json.dumps(item) for item in second_rows))
    report = audit_files([first, second])
    assert report["strict_gate_unchanged"]
    assert report["strict_failures_common_to_all_runs"] == ["instr_reverse"]
    assert report["runs"][0]["strict_passes"] == 0
    assert report["runs"][1]["strict_passes"] == 1
    assert not report["runs"][0]["complete_suite"]
    assert not report["runs"][1]["strict_gate_passed"]


def test_audit_rejects_changed_expectation(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_row("instr_first", "blue", "Blue")))
    with pytest.raises(ValueError, match="expected answer differs"):
        audit_files([path])


def test_audit_rejects_duplicate_case_and_false_strict_pass(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    row = _row("instr_first", "red", "Red")
    duplicate.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate quality case"):
        audit_files([duplicate])

    inconsistent = tmp_path / "inconsistent.jsonl"
    inconsistent.write_text(json.dumps(_row("instr_first", "red", "Red", passed=True)))
    with pytest.raises(ValueError, match="strict pass conflicts"):
        audit_files([inconsistent])
