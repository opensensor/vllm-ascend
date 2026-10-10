# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import json
import threading

import pytest

from tools.qwen4exp.benchmark_capacity import (
    capacity_summary,
    fit_prompt,
    parse_metrics,
    prompt_lengths,
    stream_request,
)


def test_asymmetric_session_lengths():
    assert prompt_lengths(256, 2, None) == [256, 256]
    assert prompt_lengths(256, 2, [256, 8192]) == [256, 8192]
    assert prompt_lengths(261632, 4, None) == [261632] * 4
    assert prompt_lengths(256, 6, None) == [256] * 6


@pytest.mark.parametrize("concurrency", [0, -1])
def test_invalid_concurrency(concurrency):
    with pytest.raises(ValueError, match="one positive prompt length"):
        prompt_lengths(256, concurrency, None)


@pytest.mark.parametrize("lengths", [[], [256], [256, 0], [-1, 8192], [256, 8192, 4096]])
def test_invalid_session_lengths(lengths):
    with pytest.raises(ValueError, match="one positive prompt length"):
        prompt_lengths(256, 2, lengths)


def test_exact_prompt_budget_keeps_both_ends():
    assert fit_prompt([1, 2], [3, 4], [5, 6], 9) == [1, 2, 3, 4, 3, 4, 3, 5, 6]
    assert fit_prompt([1], [3], [2], 2) == [1, 2]


@pytest.mark.parametrize("filler,budget", [([], 4), ([3], 1)])
def test_invalid_budget(filler, budget):
    with pytest.raises(ValueError):
        fit_prompt([1], filler, [2], budget)


def test_metrics_do_not_confuse_waiting_with_waiting_by_reason():
    result = parse_metrics(
        'vllm:num_requests_running{engine="0"} 2.0\n'
        'vllm:num_requests_waiting{engine="0"} 1.0\n'
        'vllm:num_requests_waiting_by_reason{reason="capacity"} 1.0\n'
        'vllm:kv_cache_usage_perc{engine="0"} 8.5e-1\n'
        'vllm:num_preemptions_total{engine="0"} 3.0\n'
    )
    assert result["num_requests_running"] == 2
    assert result["num_requests_waiting"] == 1
    assert result["kv_cache_usage_perc"] == 0.85
    assert result["num_preemptions_total"] == 3


def test_completed_stream_requires_exact_prompt_usage(monkeypatch):
    events = [
        {"choices": [{"text": "test output", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
    ]
    data = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n"
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(data.encode()))
    payload = {"prompt": [1, 2], "max_tokens": 3}
    result = stream_request("http://localhost", payload, threading.Barrier(1), None)
    assert result["correct"] and result["done"]
    assert result["decode_tok_s"] > 0
    with pytest.raises(AssertionError, match="prompt length"):
        stream_request("http://localhost", {**payload, "prompt": [1]}, threading.Barrier(1), None)


def test_stream_error_is_not_a_pass(monkeypatch):
    data = b'data: {"error": {"message": "engine died"}}\n\n'
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(data))
    with pytest.raises(RuntimeError, match="engine died"):
        stream_request("http://localhost", {"prompt": [1], "max_tokens": 3}, threading.Barrier(1), None)


def test_four_sessions_need_simultaneous_decode_and_no_preemption():
    results = [
        {
            "start": float(index),
            "first": 10.0 + index,
            "last": 100.0 - index,
            "end": 101.0 + index,
            "done": True,
            "correct": True,
            "usage": {"completion_tokens": 512},
        }
        for index in range(4)
    ]
    before = dict.fromkeys(
        ("num_preemptions_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"), 0.0
    )
    after = before.copy()
    peaks = {"num_requests_running": 4.0, "num_requests_waiting": 0.0, "kv_cache_usage_perc": 0.9}
    assert capacity_summary(results, peaks, before, after, 4)["passed"]
    assert not capacity_summary(results, {**peaks, "num_requests_running": 3.0}, before, after, 4)["passed"]
    assert not capacity_summary(results, peaks, before, {**after, "num_preemptions_total": 1}, 4)["passed"]
    no_overlap = [
        {**result, "first": 10.0 + index * 100, "last": 20.0 + index * 100} for index, result in enumerate(results)
    ]
    assert not capacity_summary(no_overlap, peaks, before, after, 4)["passed"]
