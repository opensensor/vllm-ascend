"""CPU-only fixtures for the parser-aware serving suite."""

from __future__ import annotations

import json

import pytest

from tools.glm_perf.suite import (
    Case,
    make_groups,
    parse_sse,
    request_body,
    retrieval_case,
    run_groups,
    run_request,
    score,
    summarize,
)


def sse(*events: dict | str) -> list[bytes]:
    return [f"data: {json.dumps(event) if isinstance(event, dict) else event}\n\n".encode() for event in events]


def response_events(
    *,
    final: str = "45",
    reasoning: str = "Compute.",
    completion_tokens: int = 5,
    finish: str = "stop",
    prompt_tokens: int = 25,
) -> list[bytes]:
    return sse(
        {"choices": [{"delta": {"reasoning_content": reasoning}}]},
        {"choices": [{"delta": {"content": final}}]},
        {"choices": [{"delta": {}, "finish_reason": finish}]},
        {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
        "[DONE]",
    )


class Stream:
    def __init__(self, lines: list[bytes]):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *_args):
        return False


def test_stream_timing_and_parser_separation() -> None:
    ticks = iter([0.0, 0.2, 0.7, 1.0])
    case = Case("answer", "arithmetic", "17 + 28?", "45", 256)
    row = run_request(
        case,
        "http://localhost:8001",
        "glm",
        42,
        "one",
        opener=lambda *_args, **_kwargs: Stream(response_events()),
        clock=lambda: next(ticks),
    )
    assert row["valid"] and row["passed"]
    assert row["content"] == "45" and row["reasoning"] == "Compute."
    assert row["ttft_s"] == pytest.approx(0.2)
    assert row["decode_s"] == pytest.approx(0.5)
    assert row["decode_tokens_per_s"] == pytest.approx(10.0)
    assert row["usage"] == {"prompt_tokens": 25, "completion_tokens": 5}
    assert row["early_eos"]
    assert row["stream_event_gap_p50_s"] == pytest.approx(0.5)
    assert row["request_settings"]["stream_options"] == {"include_usage": True}


def test_request_timeout_is_forwarded_to_transport() -> None:
    received: list[float] = []

    def opener(_request, *, timeout: float):
        received.append(timeout)
        return Stream(response_events())

    row = run_request(
        Case("answer", "arithmetic", "17 + 28?", "45", 256),
        "http://localhost:8001",
        "glm",
        42,
        "one",
        opener=opener,
        clock=lambda: 1.0,
        timeout_s=3600,
    )
    assert row["valid"] and row["passed"]
    assert received == [3600]


def test_concurrent_fairness_aggregate_and_percentiles() -> None:
    base = {
        "valid": True,
        "passed": True,
        "group_id": "four",
        "ttft_s": 0.2,
        "early_eos": False,
        "usage": {"completion_tokens": 11},
    }
    rows = [
        {**base, "first_token_s": 1.0, "last_token_s": 3.0, "decode_tokens_per_s": 5.0},
        {**base, "first_token_s": 1.2, "last_token_s": 6.2, "decode_tokens_per_s": 2.0},
    ]
    group = summarize(rows)["groups"]["four"]
    assert group["requests"] == 2
    assert group["aggregate_decode_tokens_per_s"] == pytest.approx(22 / 5.2)
    assert group["per_request_decode_p50"] == pytest.approx(3.5)
    assert group["per_request_decode_p95"] == pytest.approx(4.85)
    assert group["fairness_min_max_ratio"] == pytest.approx(0.4)


def test_early_eos_and_raw_marker_fail_quality() -> None:
    case = Case("answer", "arithmetic", "17 + 28?", "45", 256)
    parsed = parse_sse(response_events(final="</think>45"), clock=lambda: 1.0)
    assert not score(case, parsed)["passed"]
    assert score(case, parsed)["raw_thinking_marker"]
    short = Case("speed", "short", "continue", None, 256)
    assert not score(short, parsed)["passed"]
    assert score(short, parsed)["speed_sample_complete"] is False
    parsed["content"] = "45"
    parsed["usage"]["completion_tokens"] = 256
    parsed["finish_reason"] = "length"
    assert score(short, parsed)["passed"]
    fault = Case("fault_32", "fault", "continue", None, 32)
    assert score(fault, parsed)["passed"]


def test_tool_call_must_have_correct_function_and_arguments() -> None:
    tool = {"function": "get_order_status", "arguments": {"order_id": "A-1042"}}
    case = Case("tool", "tool", "Look up order", None, 256, tool=tool)
    parsed = {
        "content": "",
        "reasoning": "",
        "finish_reason": "tool_calls",
        "tool_calls": [{"id": "call_1", "name": "get_order_status", "arguments": '{"order_id": "A-1042"}'}],
    }
    assert score(case, parsed)["passed"]
    parsed["tool_calls"][0]["arguments"] = '{"order_id": "wrong"}'
    assert not score(case, parsed)["passed"]
    parsed["finish_reason"] = "stop"
    assert not score(case, parsed)["tool_call_success"]
    body = request_body(case, "glm", 42)
    assert body["tool_choice"] == "auto" and body["tools"][0]["function"]["name"] == "get_order_status"


def test_incomplete_stream_rejected() -> None:
    lines = response_events()[:-1]
    with pytest.raises(ValueError, match="incomplete stream"):
        parse_sse(lines, clock=lambda: 1.0)
    lines = response_events()[:3] + sse("[DONE]")
    with pytest.raises(ValueError, match="incomplete stream"):
        parse_sse(lines, clock=lambda: 1.0)


def test_parser_stops_at_done_without_waiting_for_socket_eof() -> None:
    def persistent_connection():
        yield from response_events()
        raise AssertionError("reader continued after [DONE]")

    parsed = parse_sse(persistent_connection(), clock=lambda: 1.0)
    assert parsed["content"] == "45"
    assert parsed["finish_reason"] == "stop"


def test_run_groups_records_each_result_in_group_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(case: Case, _base_url: str, _model: str, _seed: int, group_id: str, **_kwargs):
        return {"case_id": case.case_id, "group_id": group_id}

    monkeypatch.setattr("tools.glm_perf.suite.run_request", fake_request)
    groups = [
        ("one", [Case("a", "short", "a", None, 1)]),
        ("four", [Case(str(index), "short", "b", None, 1) for index in range(4)]),
    ]
    recorded: list[dict] = []
    rows = run_groups(groups, "http://localhost:8001", "glm", 42, on_result=recorded.append)
    assert recorded == rows
    assert [(row["group_id"], row["case_id"]) for row in rows] == [
        ("one", "a"),
        ("four", "0"),
        ("four", "1"),
        ("four", "2"),
        ("four", "3"),
    ]


def test_workload_ladder_and_fixed_quality_cases() -> None:
    groups = make_groups(["quality", "short", "retrieval", "windows", "fault", "fault4", "tool"], [32768])
    quality = [
        case
        for _name, cases in groups
        for case in cases
        if case.category in {"arithmetic", "instruction", "code", "retrieval"} and not case.target_prompt_tokens
    ]
    assert len(quality) == 20
    assert [len(cases) for name, cases in groups if name.startswith("short_")] == [1, 4]
    assert [len(cases) for name, cases in groups if name.startswith("windows_")] == [4]
    assert {name for name, _cases in groups if name in {"retrieval_8192", "retrieval_32768"}} == {
        "retrieval_8192",
        "retrieval_32768",
    }
    assert any(cases[0].max_tokens == 32 for name, cases in groups if name == "fault_32")
    assert any(
        len(cases) == 4 and all(case.max_tokens == 32 for case in cases)
        for name, cases in groups
        if name == "fault4_32"
    )


def test_retrieval_calibration_uses_token_counter_and_is_deterministic() -> None:
    counter = lambda prompt: len(prompt.split())
    first = retrieval_case(131072, category="window", variant=2, count_tokens=counter)
    second = retrieval_case(131072, category="window", variant=2, count_tokens=counter)
    filler_tokens = counter(
        "The secret code is X.\n" + "The quick brown fox jumps over the lazy dog. Keep this context in mind.\n"
    ) - counter("The secret code is X.\n")
    assert first == second
    assert first.raw_prompt_tokens == counter(first.prompt)
    assert abs(first.raw_prompt_tokens - 131072) <= filler_tokens
    assert first.expected == "BLUE-ORCHID-7319-2"


def test_server_usage_rejects_mislabeled_context_length() -> None:
    ticks = iter([0.0, 0.2, 0.7, 1.0])
    case = Case("retrieval_8192", "retrieval", "Find the code", "45", 256, 8192, raw_prompt_tokens=8190)
    row = run_request(
        case,
        "http://localhost:8001",
        "glm",
        42,
        "retrieval_8192",
        opener=lambda *_args, **_kwargs: Stream(response_events()),
        clock=lambda: next(ticks),
        tokenizer_identity={"path": "/tmp/tokenizer.json", "sha256": "abc"},
    )
    assert row["valid"] and not row["passed"]
    assert row["prompt_target_met"] is False
    assert row["prompt_token_target_error"] == 25 - 8192
    assert row["tokenizer"]["sha256"] == "abc"


def test_window_tier_reserves_completion_and_template_budget() -> None:
    counter = lambda prompt: len(prompt.split())
    groups = make_groups(["windows"], [16384], counter)
    assert len(groups) == 1
    label, cases = groups[0]
    assert label == "windows_16384"
    assert len(cases) == 4
    assert len({case.case_id for case in cases}) == 4
    assert all(case.context_tier_tokens == 16384 for case in cases)
    assert all(case.target_prompt_tokens == 16384 - 256 - 64 for case in cases)
    assert all(case.max_tokens == 256 for case in cases)
    assert all(case.raw_prompt_tokens is not None for case in cases)


def test_single_window_context_ladder_uses_one_request_per_tier() -> None:
    groups = make_groups(["windows"], [32768, 65536, 131072], window_count=1)
    assert [name for name, _cases in groups] == ["windows_32768", "windows_65536", "windows_131072"]
    assert all(len(cases) == 1 and cases[0].case_id.endswith("_0") for _name, cases in groups)
    with pytest.raises(ValueError, match="window count"):
        make_groups(["windows"], [32768], window_count=0)


def test_window_rejects_served_prompt_that_cannot_fit_completion() -> None:
    ticks = iter([0.0, 0.2, 0.7, 1.0])
    case = Case("window", "window", "Find the code", "45", 256, 16064, context_tier_tokens=16384)
    row = run_request(
        case,
        "http://localhost:8001",
        "glm",
        42,
        "windows_16384",
        opener=lambda *_args, **_kwargs: Stream(response_events(prompt_tokens=16200)),
        clock=lambda: next(ticks),
    )
    assert row["valid"] and row["prompt_target_met"]
    assert row["context_fit"] is False and not row["passed"]
