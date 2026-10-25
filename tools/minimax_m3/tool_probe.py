# SPDX-License-Identifier: Apache-2.0
"""Bounded real-model tool protocol checks, including streamed argument assembly."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def chat_request(base_url, payload, timeout=900):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    chunks, calls, text, reasoning = [], {}, [], []
    usage, finish, first = None, None, None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not payload.get("stream"):
                result = json.load(response)
                choice = result["choices"][0]
                return {
                    "message": choice["message"],
                    "finish_reason": choice["finish_reason"],
                    "usage": result.get("usage"),
                    "elapsed_seconds": time.monotonic() - started,
                    "response": result,
                    "status": response.status,
                }
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                if line[6:].strip() == b"[DONE]":
                    break
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                chunks.append(chunk)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta", {})
                    text.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                    if first is None and any(
                        delta.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls")
                    ):
                        first = time.monotonic() - started
                    for call in delta.get("tool_calls") or []:
                        target = calls.setdefault(
                            call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if call.get("id"):
                            if target["id"] and target["id"] != call["id"]:
                                raise ValueError("stream changed a tool call ID")
                            target["id"] = call["id"]
                        for field in ("name", "arguments"):
                            target["function"][field] += call.get("function", {}).get(field) or ""
            message = {"role": "assistant", "content": "".join(text), "reasoning": "".join(reasoning)}
            if calls:
                if sorted(calls) != list(range(len(calls))):
                    raise ValueError("stream tool indexes are not contiguous")
                message["tool_calls"] = [calls[i] for i in sorted(calls)]
            return {
                "message": message,
                "finish_reason": finish,
                "usage": usage,
                "chunks": chunks,
                "first_output_seconds": first,
                "elapsed_seconds": time.monotonic() - started,
                "status": response.status,
            }
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.read().decode(), "elapsed_seconds": time.monotonic() - started}


def tool(name, properties, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Return the requested information.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties) if required is None else required,
                "additionalProperties": False,
            },
        },
    }


def check_calls(result, expected, *, named=False):
    calls = result.get("message", {}).get("tool_calls", [])
    # The pinned vLLM explicitly uses `stop` for a named tool choice in both
    # full and streamed responses. Auto/required must signal `tool_calls`.
    expected_finish = "stop" if named else "tool_calls"
    if result.get("finish_reason") != expected_finish or len(calls) != len(expected):
        return False
    try:
        actual = [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in calls]
        ids = [call.get("id") for call in calls]
        return all(ids) and len(set(ids)) == len(ids) and sorted(actual, key=repr) == sorted(expected, key=repr)
    except (KeyError, json.JSONDecodeError):
        return False


def main():
    """Run exact protocol assertions, retaining raw responses for failed gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="minimax-m31-flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "auto",
            "stream",
            "named",
            "required",
            "parallel",
            "nested",
            "none",
            "json_schema",
            "json_schema_stream",
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lookup = tool("lookup_code", {"city": {"type": "string"}})
    nested = tool(
        "lookup_records",
        {
            "filters": {
                "type": "object",
                "properties": {"region": {"type": "string"}, "active": {"type": "boolean"}},
                "required": ["region", "active"],
            },
            "ids": {"type": "array", "items": {"type": "integer"}},
            "limit": {"type": "integer"},
        },
    )
    messages = [{"role": "user", "content": "Call lookup_code for city Paris. Do not guess the tool result."}]
    base = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 256,
        "tools": [lookup],
        "tool_choice": "auto",
        "messages": messages,
    }
    cases = [
        ("auto", {}, [("lookup_code", {"city": "Paris"})]),
        ("stream", {"stream": True, "stream_options": {"include_usage": True}}, [("lookup_code", {"city": "Paris"})]),
        (
            "named",
            {"tool_choice": {"type": "function", "function": {"name": "lookup_code"}}},
            [("lookup_code", {"city": "Paris"})],
        ),
        ("required", {"tool_choice": "required"}, [("lookup_code", {"city": "Paris"})]),
        (
            "parallel",
            {
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [
                    {
                        "role": "user",
                        "content": "Call lookup_code twice in one turn: city Paris and city London. "
                        "Do not guess results.",
                    }
                ],
            },
            [("lookup_code", {"city": "Paris"}), ("lookup_code", {"city": "London"})],
        ),
        (
            "nested",
            {
                "stream": True,
                "stream_options": {"include_usage": True},
                "tools": [nested],
                "messages": [
                    {
                        "role": "user",
                        "content": 'Call lookup_records with filters {"region":"east","active":true}, '
                        "ids [3,7], and limit 2. Do not guess results.",
                    }
                ],
            },
            [("lookup_records", {"filters": {"region": "east", "active": True}, "ids": [3, 7], "limit": 2})],
        ),
    ]
    reports = []
    for name, extra, expected in cases:
        if args.case and name not in args.case:
            continue
        result = chat_request(args.base_url, {**base, **extra})
        result.update(name=name, passed=check_calls(result, expected, named=name == "named"))
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        reports.append({key: value for key, value in result.items() if key not in ("chunks", "response")})
        print(json.dumps(reports[-1]), flush=True)
        if name == "auto" and result["passed"]:
            call = result["message"]["tool_calls"][0]
            follow = [
                *messages,
                {"role": "assistant", "content": result["message"].get("content"), "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": '{"code":"cedar-47"}'},
                {"role": "user", "content": "Reply with only the code returned by the tool."},
            ]
            answer = chat_request(
                args.base_url, {**base, "messages": follow, "stream": True, "stream_options": {"include_usage": True}}
            )
            answer.update(
                name="roundtrip",
                passed=answer.get("message", {}).get("content", "").strip() == "cedar-47"
                and not answer.get("message", {}).get("tool_calls")
                and answer.get("finish_reason") == "stop",
            )
            (args.output / "roundtrip.json").write_text(json.dumps(answer, indent=2) + "\n")
            reports.append({key: value for key, value in answer.items() if key not in ("chunks", "response")})
            print(json.dumps(reports[-1]), flush=True)
    extra_cases = [
        (
            "none",
            {
                **base,
                "tool_choice": "none",
                "messages": [{"role": "user", "content": "Do not use any tools. Reply exactly ready."}],
                "max_tokens": 256,
            },
        ),
        (
            "json_schema",
            {
                "model": args.model,
                "temperature": 0,
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "What is 17 + 25? Return the answer in the result field."}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"result": {"type": "integer"}},
                            "required": ["result"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        ),
    ]
    extra_cases.append(
        ("json_schema_stream", {**extra_cases[-1][1], "stream": True, "stream_options": {"include_usage": True}})
    )
    for name, payload in extra_cases:
        if args.case and name not in args.case:
            continue
        result = chat_request(args.base_url, payload)
        message = result.get("message", {})
        try:
            content = message.get("content") or ""
            passed = (
                content.strip().lower().rstrip(".!") == "ready"
                if name == "none"
                else json.loads(content) == {"result": 42}
            )
            passed = passed and not message.get("tool_calls") and result.get("finish_reason") == "stop"
        except json.JSONDecodeError:
            passed = False
        result.update(name=name, passed=passed)
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        reports.append({key: value for key, value in result.items() if key not in ("chunks", "response")})
        print(json.dumps(reports[-1]), flush=True)
    report = {"cases": reports, "passed": all(case["passed"] for case in reports), "count": len(reports)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
