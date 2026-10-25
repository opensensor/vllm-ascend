# SPDX-License-Identifier: Apache-2.0
"""Verify tool-call assembly rather than accepting nonempty HTTP output."""

import io
import json
from unittest.mock import patch

import pytest

from tools.minimax_m3.build_resident_native import build
from tools.minimax_m3.tool_probe import chat_request, check_calls, tokenize_chat, tool


class Response(io.BytesIO):
    status = 200


@pytest.mark.parametrize("canonical,expected", [(None, "prior thought"), ("preferred", "preferred"), ("", "")])
def test_tokenization_preserves_chat_reasoning_without_mutating_history(canonical, expected):
    payload = {
        "model": "minimax",
        "max_tokens": 256,
        "messages": [
            {"role": "assistant", "content": None, "reasoning_content": "prior thought", "reasoning": canonical}
        ],
        "tools": [tool("lookup", {})],
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    original = json.dumps(payload)
    with patch("urllib.request.urlopen", return_value=Response(b'{"count": 42, "tokens": [1, 2]}')) as send:
        assert tokenize_chat("http://localhost", payload)["count"] == 42
    request = send.call_args.args[0]
    body = json.loads(request.data)
    assert request.full_url == "http://localhost/tokenize"
    assert body["messages"][0]["reasoning"] == expected
    assert "reasoning_content" not in body["messages"][0]
    assert body["tools"] == payload["tools"]
    assert body["chat_template_kwargs"] == payload["chat_template_kwargs"]
    assert "max_tokens" not in body
    assert json.dumps(payload) == original


def test_tool_schema_can_leave_every_argument_optional():
    definition = tool("lookup", {"city": {"type": "string"}}, required=[])
    assert definition["function"]["parameters"]["required"] == []


def sse(deltas):
    events = [{"choices": [{"delta": delta, "finish_reason": None}]} for delta in deltas]
    events.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    events.append({"choices": [], "usage": {"completion_tokens": 8}})
    return Response(
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
    )


def test_streamed_calls_preserve_ids_and_nested_json_across_fragments():
    response = sse(
        [
            {"reasoning": "Use tools."},
            {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "lookup", "arguments": '{"ids":['}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '3,7],"active":true}'}}]},
            {
                "tool_calls": [
                    {"index": 1, "id": "call_b", "function": {"name": "lookup", "arguments": '{"city":"Montréal"}'}}
                ]
            },
        ]
    )
    with patch("urllib.request.urlopen", return_value=response):
        result = chat_request("http://localhost", {"stream": True})
    assert check_calls(result, [("lookup", {"ids": [3, 7], "active": True}), ("lookup", {"city": "Montréal"})])
    assert result["message"]["reasoning"] == "Use tools."
    assert result["usage"]["completion_tokens"] == 8


def test_changed_stream_call_id_is_rejected():
    response = sse(
        [
            {"tool_calls": [{"index": 0, "id": "first", "function": {"name": "lookup"}}]},
            {"tool_calls": [{"index": 0, "id": "different", "function": {"arguments": "{}"}}]},
        ]
    )
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(ValueError, match="changed"):
        chat_request("http://localhost", {"stream": True})


@pytest.mark.parametrize(
    "arguments,finish", [('"not an object"', "tool_calls"), ('{"x":2}', "stop"), ('{"x":', "tool_calls")]
)
def test_wrong_arguments_or_finish_reason_fail(arguments, finish):
    result = {
        "finish_reason": finish,
        "message": {"tool_calls": [{"id": "call_a", "function": {"name": "lookup", "arguments": arguments}}]},
    }
    assert not check_calls(result, [("lookup", {"x": 2})])


def test_named_choice_uses_pinned_vllm_stop_convention():
    result = {
        "finish_reason": "stop",
        "message": {"tool_calls": [{"id": "call_a", "function": {"name": "lookup", "arguments": '{"x":2}'}}]},
    }
    assert check_calls(result, [("lookup", {"x": 2})], named=True)
    assert not check_calls(result, [("lookup", {"x": 2})])


def test_parallel_calls_require_distinct_ids():
    call = {"id": "call_a", "function": {"name": "lookup", "arguments": '{"x":2}'}}
    result = {"finish_reason": "tool_calls", "message": {"tool_calls": [call, call]}}
    assert not check_calls(result, [("lookup", {"x": 2}), ("lookup", {"x": 2})])


def test_stream_indexes_must_be_contiguous():
    response = sse([{"tool_calls": [{"index": 1, "id": "a", "function": {"name": "lookup", "arguments": "{}"}}]}])
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(ValueError, match="contiguous"):
        chat_request("http://localhost", {"stream": True})


@pytest.mark.parametrize("enabled,max_k", [(False, 6144), (True, 8192)])
def test_native_build_rejects_unqualified_l1_geometry_before_writing(tmp_path, enabled, max_k):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="L1 max K"):
        build(output, tmp_path, "minimax_w4_test", tmp_path, singleton_l1=enabled, l1_max_k=max_k)
    assert not output.exists()


def test_native_build_rejects_conflicting_l1_schedules(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="not both"):
        build(output, tmp_path, "minimax_w4_test", tmp_path, singleton_l1=True, wide_l1=True)
    assert not output.exists()


def test_native_build_rejects_scale_cache_without_wide_l1(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="scale cache requires wide L1"):
        build(output, tmp_path, "minimax_w4_test", tmp_path, scale_cache=True)
    assert not output.exists()


@pytest.mark.parametrize("lanes,wide", [(3, True), (16, True), (4, False)])
def test_native_build_rejects_invalid_expert_teams(tmp_path, lanes, wide):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="expert lanes"):
        build(output, tmp_path, "minimax_w4_test", tmp_path, expert_lanes=lanes, wide_l1=wide)
    assert not output.exists()
