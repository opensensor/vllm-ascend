# SPDX-License-Identifier: Apache-2.0
"""Portable acceptance protocol, privacy, precision and lifecycle regressions."""

import copy
import hashlib
import json

import pytest

from tools.sidecard import execution_acceptance as suite


def cases():
    return json.loads(suite.PUBLIC_CASES.read_bytes())["acceptance_cases"]


def answer(case, *, reasoning="thinking", count=4):
    score = case["score"]
    message = {"reasoning_content": reasoning, "content": ""}
    if score["kind"] == "tool":
        message["tool_calls"] = [{"function": {"name": score["name"], "arguments": json.dumps(score["arguments"])}}]
    elif score["kind"] in {"json"}:
        message["content"] = json.dumps(score["expected"])
    elif score["kind"] != "capacity":
        message["content"] = score.get("expected", "needle")
    prompt = case.get("expected_prompt_tokens", 30)
    return {
        "choices": [
            {"message": message, "index": 0, "finish_reason": "tool_calls" if score["kind"] == "tool" else "stop"}
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": count, "total_tokens": prompt + count},
    }


def capacity():
    body = {
        "prompt": [1] * 131008,
        "max_tokens": 64,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return {
        "name": "capacity",
        "tier": "context",
        "category": "capacity",
        "request": body,
        "request_sha256": suite.canonical_sha256(body),
        "expected_prompt_tokens": 131008,
        "score": {"kind": "capacity"},
    }


def reference(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def manifest(root, acceptance=None):
    identity = {name: "a" * 64 for name in ("source_sha256", "config_sha256", "checkpoint_sha256", "native_sha256")}
    identity.update(
        parser="glm53", buffering_policy="separated", architecture="cuda-backbone-ascend-experts", max_model_len=131072
    )
    contract = {
        "identity": identity,
        "launch_id": "launch-final",
        "launch_started_unix_seconds": 5,
        "prefix_cache_enabled": False,
        "startup_gates_passed": True,
        "warmup_completed": True,
        "warmup_policy": "fixed",
    }
    run_ref = reference(root / "run.json", {"performance_contract": contract})
    value = {
        "schema_version": 1,
        "model": "glm-sidecard",
        "fixtures": [],
        "service": {
            "identity": identity,
            "run_manifest": run_ref,
            "startup_gates": {"passed": True, "evidence": "gates"},
            "warmup": {"completed": True, "policy": "fixed", "evidence": "warmup"},
            "cache_policy": {"mode": "disabled", "evidence": "config"},
        },
    }
    if acceptance is not None:
        value["acceptance_cases"] = acceptance
    return value


def test_public_twenty_case_coverage_and_scores():
    assert len(cases()) == 20
    assert {case["category"] for case in cases()} == suite.CATEGORIES
    for case in cases():
        suite.validate_case(case, "glm-sidecard")
        assert suite.completed_result(answer(case), case)["passed"]


@pytest.mark.parametrize(
    "change,flag",
    [
        ("reasoning_only", "incorrect_final_answer"),
        ("length", "unfinished_answer"),
        ("no_finish", "unfinished_answer"),
        ("no_usage", "invalid_usage"),
        ("zero_output", "invalid_usage"),
        ("wrong_prompt", "prompt_identity_mismatch"),
        ("bad_total", "invalid_total_usage"),
        ("leaked_reasoning", "incorrect_final_answer"),
        ("over_budget", "invalid_usage"),
        ("missing_done", "incomplete_response"),
    ],
)
def test_admission_reasoning_prefix_and_partial_stream_never_qualify(change, flag):
    case = cases()[0]
    case["expected_prompt_tokens"] = 30
    response = answer(case)
    if change == "reasoning_only":
        response["choices"][0]["message"]["content"] = ""
    elif change in {"length", "no_finish"}:
        response["choices"][0]["finish_reason"] = "length" if change == "length" else None
    elif change == "no_usage":
        response.pop("usage")
    elif change == "zero_output":
        response["usage"]["completion_tokens"] = 0
    elif change == "wrong_prompt":
        response["usage"]["prompt_tokens"] = 31
    elif change == "bad_total":
        response["usage"]["total_tokens"] = 100
    elif change == "leaked_reasoning":
        response["choices"][0]["message"]["content"] = "<think>42</think>42"
    elif change == "over_budget":
        response["usage"]["completion_tokens"] = 8193
    result = suite.completed_result(response, case, done=change != "missing_done")
    assert not result["passed"] and flag in result["flags"]


def test_completed_wrong_answer_distinct_from_unfinished():
    case = cases()[0]
    response = answer(case)
    response["choices"][0]["message"]["content"] = "41"
    result = suite.completed_result(response, case)
    assert result["completed"] and result["quality_passed"] is False


def test_capacity_requires_all_64_tokens_and_exact_prefill():
    case = capacity()
    response = answer(case, count=64)
    response["choices"][0]["finish_reason"] = "length"
    result = suite.completed_result(response, case)
    assert result["capacity_passed"] and result["quality_passed"] is None
    for field, value in (("prompt_tokens", 131007), ("completion_tokens", 63)):
        changed = copy.deepcopy(response)
        changed["usage"][field] = value
        assert not suite.completed_result(changed, case)["capacity_passed"]
    assert not suite.completed_result(response, case, done=False)["passed"]


def test_long_stream_preflight_requires_usage_before_costly_prefill():
    case = capacity()
    case["request"].pop("stream_options")
    case["request_sha256"] = suite.canonical_sha256(case["request"])
    with pytest.raises(ValueError, match="completion usage"):
        suite.validate_case(case, "glm-sidecard")


@pytest.mark.parametrize("kind", ["name", "arguments", "malformed", "missing", "two"])
def test_tool_scoring_validates_actual_final_arguments(kind):
    case = cases()[-1]
    response = answer(case)
    calls = response["choices"][0]["message"]["tool_calls"]
    if kind == "name":
        calls[0]["function"]["name"] = "other"
    elif kind in {"arguments", "malformed"}:
        calls[0]["function"]["arguments"] = "{}" if kind == "arguments" else "{"
    elif kind == "missing":
        calls.clear()
    else:
        calls.append(copy.deepcopy(calls[0]))
    assert not suite.completed_result(response, case)["quality_passed"]


def test_code_is_parsed_without_executing_model_output():
    case = cases()[8]
    message = {"content": "```python\ndef add(a,b): return a+b\n```"}
    assert suite.score_answer(message, case["score"])
    assert not suite.score_answer({"content": "import os\nos.system('touch /tmp/unsafe')"}, case["score"])


def test_contains_has_positive_and_negative_checks():
    score = {"kind": "contains", "required": ["needle"], "forbidden": ["wrong"]}
    assert suite.score_answer({"content": "the needle"}, score)
    assert not suite.score_answer({"content": "needle wrong"}, score)


def test_stream_preserves_reasoning_final_and_usage():
    events = [
        {"choices": [{"delta": {"reasoning_content": "think"}}]},
        {"choices": [{"delta": {"content": "4"}}]},
        {"choices": [{"delta": {"content": "2"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}},
    ]
    lines = [b"data: " + json.dumps(event).encode() + b"\n" for event in events] + [b"data: [DONE]\n"]
    response, done = suite.read_stream(lines)
    assert suite.completed_result(response, cases()[0], done=done)["passed"]
    assert response["choices"][0]["message"]["reasoning_content"] == "think"
    _, done = suite.read_stream(lines[:-1])
    assert not done
    with pytest.raises(ValueError, match="after DONE"):
        suite.read_stream(lines + lines[:1])


def test_stream_tools_are_reassembled_before_json_scoring():
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "record_value", "arguments": '{"name":"sample3",'}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"value":14}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}},
    ]
    lines = [b"data: " + json.dumps(event).encode() for event in events] + [b"data: [DONE]"]
    response, done = suite.read_stream(lines)
    assert suite.completed_result(response, cases()[-1], done=done)["passed"]


@pytest.mark.parametrize("change", ["hash", "reserve", "tools", "capacity", "tier", "model"])
def test_rejects_changed_input_contract(change):
    case = cases()[0]
    if change == "hash":
        case["request"]["messages"][0]["content"] = "shortened"
    elif change == "reserve":
        case["request"]["max_tokens"] = 32
        case["request_sha256"] = suite.canonical_sha256(case["request"])
    elif change == "tools":
        case["expected_tool_count"] = 23
    elif change == "capacity":
        case = capacity()
        case["request"]["ignore_eos"] = False
        case["request_sha256"] = suite.canonical_sha256(case["request"])
    elif change == "tier":
        case["tier"] = "bogus"
    else:
        case["request"]["model"] = "other"
        case["request_sha256"] = suite.canonical_sha256(case["request"])
    with pytest.raises(ValueError):
        suite.validate_case(case, "glm-sidecard")


def test_private_original_fixture_tools_and_prompt_remain_authenticated(tmp_path):
    value = manifest(tmp_path)
    body = {"prompt": "private original", "tools": [{"name": "a"}]}
    value["fixtures"] = [{"request": body, "request_sha256": suite.canonical_sha256(body), "expected_tool_count": 23}]
    with pytest.raises(ValueError, match="tools"):
        suite.validate_manifest(value, tmp_path)
    value["fixtures"][0]["expected_tool_count"] = 1
    body["prompt"] = "compacted"
    with pytest.raises(ValueError, match="fixture changed"):
        suite.validate_manifest(value, tmp_path)


def test_quality_run_records_scores_without_private_prompts_or_outputs(tmp_path):
    value = manifest(tmp_path)
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(value))
    report = suite.run(
        "http://offline",
        path,
        tmp_path / "quality.json",
        "quality",
        sender=lambda _, case, __: suite.completed_result(answer(case), case),
    )
    assert report["passed"] and report["completed_requests"] == 20
    assert not report["hardware_validated"]
    raw = (tmp_path / "quality.json").read_text()
    assert "What is 17" not in raw and "thinking" not in raw and "sample3" not in raw
    with pytest.raises(FileExistsError):
        suite.run("http://offline", path, tmp_path / "quality.json", "quality")


def test_lifecycle_missing_checks_are_pending_not_passed(tmp_path):
    rows = suite.lifecycle_results(manifest(tmp_path), tmp_path)
    assert len(rows) == 5 and all(row["status"] == "pending" and not row["passed"] for row in rows)


@pytest.mark.parametrize("kind", ["prefix", "state-reset", "cancellation"])
def test_owner_lifecycle_requires_real_completed_response_proofs(tmp_path, kind):
    value = manifest(tmp_path)
    case = cases()[0]
    response = answer(case)
    response["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
    proof = {"case": case, "response": response, "done": True}
    receipt = {
        "kind": kind,
        "run_manifest_sha256": value["service"]["run_manifest"]["sha256"],
        "passed": True,
        "completed": True,
        "evidence": "local owner verified request retirement and slot leases",
        "checks": dict.fromkeys(suite.LIFECYCLE_CHECKS[kind], True),
    }
    if kind == "cancellation":
        receipt["post_action"] = proof
    else:
        receipt["completed_requests"] = [copy.deepcopy(proof), copy.deepcopy(proof)]
        if kind == "state-reset":
            other = cases()[1]
            receipt["completed_requests"].insert(1, {"case": other, "response": answer(other), "done": True})
    value["lifecycle_receipts"] = [reference(tmp_path / f"{kind}.json", receipt)]
    row = next(row for row in suite.lifecycle_results(value, tmp_path) if row["name"] == kind)
    assert row["passed"]
    target = receipt["post_action"] if kind == "cancellation" else receipt["completed_requests"][-1]
    target["done"] = False
    value["lifecycle_receipts"] = [reference(tmp_path / f"{kind}.json", receipt)]
    row = next(row for row in suite.lifecycle_results(value, tmp_path) if row["name"] == kind)
    assert not row["passed"]


def test_checkbox_only_prefix_receipt_cannot_pass(tmp_path):
    value = manifest(tmp_path)
    receipt = {
        "kind": "prefix",
        "run_manifest_sha256": value["service"]["run_manifest"]["sha256"],
        "passed": True,
        "completed": True,
        "evidence": "claimed successful",
        "checks": dict.fromkeys(suite.LIFECYCLE_CHECKS["prefix"], True),
    }
    value["lifecycle_receipts"] = [reference(tmp_path / "prefix.json", receipt)]
    row = next(row for row in suite.lifecycle_results(value, tmp_path) if row["name"] == "prefix")
    assert not row["passed"]


def test_stability_runs_100_mixed_requests_without_faking_restart(tmp_path, monkeypatch):
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(manifest(tmp_path)))
    monkeypatch.setattr(suite, "over_limit_result", lambda *_: {"name": "over-limit", "passed": True, "status": 400})
    report = suite.run(
        "http://offline",
        path,
        tmp_path / "stability.json",
        "stability",
        sender=lambda _, case, __: suite.completed_result(answer(case), case),
    )
    assert report["generation_passed"] and report["completed_requests"] == 100
    assert not report["passed"] and set(report["pending"]) == set(suite.LIFECYCLE_CHECKS)
    assert {row["category"] for row in report["cases"]} == suite.CATEGORIES


def test_restart_requires_authenticated_distinct_launch_and_completed_answer(tmp_path):
    value = manifest(tmp_path)
    after = value["service"]["run_manifest"]
    before_value = json.loads((tmp_path / "run.json").read_text())
    before_value["performance_contract"].update(launch_id="before", launch_started_unix_seconds=1)
    before = reference(tmp_path / "before.json", before_value)
    case = cases()[0]
    receipt = {
        "kind": "restart",
        "run_manifest_sha256": after["sha256"],
        "passed": True,
        "completed": True,
        "evidence": "owner transition log",
        "checks": dict.fromkeys(suite.LIFECYCLE_CHECKS["restart"], True),
        "before_run_manifest": before,
        "after_run_manifest": after,
        "post_action": {"case": case, "response": answer(case), "done": True},
    }
    value["lifecycle_receipts"] = [reference(tmp_path / "restart.json", receipt)]
    assert suite.lifecycle_results(value, tmp_path)[0]["passed"]
    receipt["post_action"]["done"] = False
    value["lifecycle_receipts"] = [reference(tmp_path / "restart.json", receipt)]
    assert not suite.lifecycle_results(value, tmp_path)[0]["passed"]
    receipt["before_run_manifest"] = after
    value["lifecycle_receipts"] = [reference(tmp_path / "restart.json", receipt)]
    assert not suite.lifecycle_results(value, tmp_path)[0]["passed"]


def test_context_rejects_admission_only_or_missing_long_retrieval(tmp_path):
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(manifest(tmp_path, [capacity()])))
    with pytest.raises(ValueError, match="retrieval"):
        suite.run("http://offline", path, tmp_path / "context.json", "context")


def test_exact_32k_128k_retrieval_and_capacity_are_consumed(tmp_path):
    long_cases = []
    for size in (32768, 131072):
        body = {"prompt": [1] * (size - 8192), "max_tokens": 8192}
        long_cases.append(
            {
                "name": f"retrieval-{size}",
                "tier": "context",
                "category": "retrieval",
                "context_tier": size,
                "request": body,
                "request_sha256": suite.canonical_sha256(body),
                "expected_prompt_tokens": size - 8192,
                "score": {"kind": "exact", "expected": "needle"},
            }
        )
    long_cases.append(capacity())
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(manifest(tmp_path, long_cases)))

    def sender(_, case, __):
        response = answer(case, count=64 if case["category"] == "capacity" else 4)
        if case["category"] == "capacity":
            response["choices"][0]["finish_reason"] = "length"
        return suite.completed_result(response, case)

    report = suite.run("http://offline", path, tmp_path / "context.json", "context", sender=sender)
    assert report["passed"] and report["completed_requests"] == 3
    assert report["cases"][-1]["capacity_passed"]


def test_request_failure_is_recorded_and_receipt_written(tmp_path):
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(manifest(tmp_path)))

    def fail(*_):
        raise OSError("private request text must not appear")

    report = suite.run("http://offline", path, tmp_path / "failed.json", "quality", sender=fail)
    assert not report["passed"] and report["completed_requests"] == 0
    assert "private request" not in (tmp_path / "failed.json").read_text()


@pytest.mark.parametrize("base", ["http://host:8000", "http://host:8000/v1/"])
def test_base_url_normalization(base):
    assert suite.endpoint(base, {"messages": []}) == "http://host:8000/v1/chat/completions"
