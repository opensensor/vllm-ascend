# SPDX-License-Identifier: Apache-2.0
import io
import json

import pytest

from tools.sidecard.prefill_benchmark import exact_prompt, stream_measure
from tools.sidecard.service_validate import run as validate_service


def test_exact_prompt():
    assert exact_prompt([1, 2, 3], 5) == [1, 2, 3, 1, 2]
    with pytest.raises(ValueError):
        exact_prompt([], 5)
    with pytest.raises(ValueError):
        exact_prompt([1], 0)


def test_completed_stream_and_cache(monkeypatch):
    events = [
        dict(choices=[dict(text="")]),
        dict(choices=[dict(text="four")]),
        dict(choices=[], usage=dict(prompt_tokens=4, completion_tokens=1, prompt_tokens_details=dict(cached_tokens=2))),
    ]
    raw = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(raw + b"data: [DONE]\n"))
    result = stream_measure("http://localhost", [1, 2, 3, 4])
    assert result["cached_tokens"] == 2
    assert result["text"] == "four"
    assert result["generation_completed"]


def test_missing_usage_fails(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(b'data: {"choices":[{"text":"four"}]}\ndata: [DONE]\n'),
    )
    with pytest.raises(AssertionError, match="token usage"):
        stream_measure("http://localhost", [1])


def test_invalid_validation_prefill_budget(tmp_path):
    with pytest.raises(ValueError, match="positive prefill budget"):
        validate_service("http://localhost", tmp_path / "metrics", tmp_path / "report", prefill_budget=0)
