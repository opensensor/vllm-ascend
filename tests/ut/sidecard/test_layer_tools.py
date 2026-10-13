# SPDX-License-Identifier: Apache-2.0
"""Checkpoint XML tool calls, fragmented streams, and OpenAI history regressions."""

import json

import pytest

from tools.sidecard.layer_server import request_tokens, text_messages
from tools.sidecard.layer_tools import Tools, ToolStream, tool_message


def definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}, "offset": {"type": "integer"}},
                    "required": ["filePath"],
                    "additionalProperties": False,
                },
            },
        }
    ]


BODY = (
    "read<arg_key>filePath</arg_key><arg_value> /tmp/file </arg_value><arg_key>offset</arg_key><arg_value>3</arg_value>"
)
TEXT = "reason</think>Checking.<tool_call>" + BODY + "</tool_call>"


def test_tools_pass_definitions_to_checkpoint_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"] == definitions()
            return [1, 2]

    assert request_tokens(
        {"messages": [{"role": "user", "content": "Read it"}], "tools": definitions(), "max_tokens": 2},
        Tokenizer(),
        8,
        chat=True,
    ) == ([1, 2], 2)


def test_none_choice_omits_schemas_from_prompt_and_parser():
    tools = Tools({"tools": definitions(), "tool_choice": "none"})
    assert not tools.definitions and not tools.schemas
    assert ToolStream(tools).feed(TEXT, final=True)["content"] == "Checking.<tool_call>" + BODY + "</tool_call>"


def test_nonstream_response_has_typed_arguments_and_no_tool_markup():
    message = tool_message(TEXT, Tools({"tools": definitions()}))
    assert message["content"] == "Checking." and message["reasoning_content"] == "reason"
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"filePath": " /tmp/file ", "offset": 3}


@pytest.mark.parametrize("split", range(len(TEXT) + 1))
def test_every_stream_boundary_reassembles_one_valid_call(split):
    stream = ToolStream(Tools({"tools": definitions()}))
    chunks = [stream.feed(TEXT[:split]), stream.feed(TEXT[split:]), stream.feed("", final=True)]
    assert "".join(v.get("content", "") for v in chunks) == "Checking."
    assert "".join(v.get("reasoning_content", "") for v in chunks) == "reason"
    calls = [c for v in chunks for c in v.get("tool_calls", [])]
    assert len(calls) == 1 and calls[0]["index"] == 0
    assert json.loads(calls[0]["function"]["arguments"]) == {"filePath": " /tmp/file ", "offset": 3}


def test_character_stream_and_multiple_calls_keep_stable_order():
    stream = ToolStream(Tools({"tools": definitions()}))
    text = TEXT + "<tool_call>" + BODY.replace(" /tmp/file ", "second") + "</tool_call>Done"
    chunks = [stream.feed(character) for character in text] + [stream.feed("", final=True)]
    calls = [c for v in chunks for c in v.get("tool_calls", [])]
    assert [c["index"] for c in calls] == [0, 1]
    assert len({c["id"] for c in calls}) == 2
    assert "".join(v.get("content", "") for v in chunks) == "Checking.Done"


@pytest.mark.parametrize(
    "block",
    [
        BODY.replace("read", "unknown", 1),
        BODY.replace("<arg_value>3</arg_value>", "<arg_value>false</arg_value>"),
        "read",
        BODY + "<arg_key>offset</arg_key><arg_value>4</arg_value>",
        BODY + "<arg_key>extra</arg_key><arg_value>x</arg_value>",
        BODY.replace("</arg_value>", "", 1),
    ],
)
def test_invalid_generated_calls_are_visible_text_and_never_dispatched(block):
    stream = ToolStream(Tools({"tools": definitions()}))
    text = "</think><tool_call>" + block + "</tool_call>"
    result = stream.feed(text, final=True)
    assert not stream.calls and not result.get("tool_calls")
    assert result["content"] == "<tool_call>" + block + "</tool_call>"


def test_truncated_call_never_becomes_executable():
    stream = ToolStream(Tools({"tools": definitions()}))
    assert stream.feed("</think><tool_call>" + BODY) == {}
    result = stream.feed("", final=True)
    assert not stream.calls and result["content"] == "<tool_call>" + BODY


def test_assistant_calls_and_tool_results_round_trip_without_mutating_input():
    message = tool_message(TEXT, Tools({"tools": definitions()}))
    history = [
        {"role": "user", "content": "Read"},
        message,
        {
            "role": "tool",
            "tool_call_id": message["tool_calls"][0]["id"],
            "content": [{"type": "text", "text": "contents"}],
        },
    ]
    normalized = text_messages(history)
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"filePath": " /tmp/file ", "offset": 3}
    assert isinstance(history[1]["tool_calls"][0]["function"]["arguments"], str)
    assert normalized[2]["content"] == "contents"
    message["content"] = None
    assert text_messages(history)[1]["content"] == ""


@pytest.mark.parametrize(
    "tools", [None, [{}], [{"type": "function", "function": {"name": "bad name"}}], definitions() * 2]
)
def test_invalid_client_tools_fail_admission(tools):
    with pytest.raises(ValueError):
        Tools({"tools": tools})


def test_history_rejects_foreign_or_duplicate_tool_results():
    message = tool_message(TEXT, Tools({"tools": definitions()}))
    result = {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "contents"}
    with pytest.raises(ValueError):
        text_messages([result])
    with pytest.raises(ValueError):
        text_messages([message, result, result])


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "read"}}])
def test_unsupported_forced_choices_are_explicit(choice):
    with pytest.raises(ValueError, match="tool_choice"):
        Tools({"tools": definitions(), "tool_choice": choice})


def test_client_schema_never_fetches_remote_references():
    tools = definitions()
    tools[0]["function"]["parameters"]["properties"]["filePath"] = {"$ref": "https://example.invalid/schema"}
    with pytest.raises(ValueError, match="local"):
        Tools({"tools": tools})


def test_unresolved_local_reference_never_dispatches_a_call():
    tools = definitions()
    tools[0]["function"]["parameters"]["properties"]["filePath"] = {"$ref": "#/$defs/missing"}
    stream = ToolStream(Tools({"tools": tools}))
    result = stream.feed(TEXT, final=True)
    assert not stream.calls and not result.get("tool_calls")


@pytest.mark.parametrize(
    "kind,raw,expected",
    [
        ("boolean", "true", True),
        ("number", "1.25", 1.25),
        ("null", "null", None),
        ("array", '[1, "two"]', [1, "two"]),
        ("object", '{"nested": [false, 3]}', {"nested": [False, 3]}),
    ],
)
def test_native_argument_values_follow_registered_json_types(kind, raw, expected):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": kind}},
                    "required": ["value"],
                },
            },
        }
    ]
    call = Tools({"tools": tools}).parse("inspect<arg_key>value</arg_key><arg_value>" + raw + "</arg_value>")
    assert json.loads(call["function"]["arguments"]) == {"value": expected}
