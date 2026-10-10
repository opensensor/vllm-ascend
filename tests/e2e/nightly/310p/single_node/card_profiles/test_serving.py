# SPDX-License-Identifier: Apache-2.0
"""Real-weight serving and prefix-reuse gates, not dummy equivalence claims."""

import json
import time


def test_text_answer_and_request_cache_accounting(glm_profile_server, record_property):
    selected, call = glm_profile_server
    # Enough text for multiple aligned cache blocks; no cache reset or worker RPC.
    context = "Static validation material for prefix cache reuse. " * 300
    payload = {
        "model": selected.name,
        "messages": [{"role": "user", "content": context + "\nWhat is 17 multiplied by 23? Answer only the number."}],
        "temperature": 0,
        "max_tokens": 64,
        "thinking_token_budget": 0,
    }
    results = []
    for ordinal in range(2):
        started = time.monotonic()
        result = call("/v1/chat/completions", payload)
        record_property(f"request_{ordinal}_elapsed_s", time.monotonic() - started)
        assert "391" in result["choices"][0]["message"]["content"]
        assert result["usage"]["completion_tokens"] > 0
        assert "cached_tokens" in result["usage"]["prompt_tokens_details"]
        results.append(result)
    cached = results[1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0, "repeated long prefix should reuse aligned history and recurrent state"
    assert results[0]["usage"]["prompt_tokens"] == results[1]["usage"]["prompt_tokens"]
    record_property("repeat_cached_tokens", cached)


def test_coding_client_tool_call_arguments(glm_profile_server):
    selected, call = glm_profile_server
    result = call(
        "/v1/chat/completions",
        {
            "model": selected.name,
            "messages": [
                {"role": "user", "content": "Call multiply with a=17 and b=23. Use the tool rather than answering."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "description": "Multiply two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 128,
            "thinking_token_budget": 0,
        },
    )
    tools = result["choices"][0]["message"]["tool_calls"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "multiply"
    assert json.loads(tools[0]["function"]["arguments"]) == {"a": 17, "b": 23}
