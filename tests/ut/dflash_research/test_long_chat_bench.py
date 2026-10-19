import pytest

from tools.dflash_research.long_chat_bench import build_payload, extract_completion


def test_chat_request_uses_natural_stopping_and_disables_thinking():
    payload = build_payload("Hello", 1536)
    assert payload["max_tokens"] == 1536
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "ignore_eos" not in payload
    assert payload["messages"][-1]["content"] == "Hello"


def test_extract_completion_preserves_finish_reason_and_timings():
    response = {
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        "timings": {"predicted_per_second": 90.0},
    }
    parsed = extract_completion(response)
    assert parsed["content"] == "answer"
    assert parsed["finish_reason"] == "stop"
    assert parsed["completion_tokens"] == 4
    assert parsed["timings"]["predicted_per_second"] == 90.0


def test_extract_completion_rejects_empty_output():
    response = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 0},
        "timings": {},
    }
    with pytest.raises(ValueError, match="no completion tokens"):
        extract_completion(response)
