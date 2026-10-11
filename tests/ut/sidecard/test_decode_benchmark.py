# SPDX-License-Identifier: Apache-2.0
import io
import json

import pytest

from tools.sidecard import decode_benchmark, prefill_benchmark


def test_decode_rate_excludes_prefill_and_first_token(monkeypatch):
    times = iter((10.0, 12.0, 18.0))
    monkeypatch.setattr(prefill_benchmark.time, "monotonic", lambda: next(times))
    events = [
        dict(choices=[dict(text="generated")]),
        dict(choices=[], usage=dict(prompt_tokens=2, completion_tokens=64)),
    ]
    raw = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n"

    def open_request(request, **kwargs):
        body = json.loads(request.data)
        assert body["max_tokens"] == 64 and body["ignore_eos"] is True
        return io.BytesIO(raw)

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    result = prefill_benchmark.stream_measure("http://localhost", [1, 2], max_tokens=64, ignore_eos=True)
    assert result["time_to_first_text_seconds"] == 2
    assert result["decode_tokens_per_second"] == 10.5


@pytest.mark.parametrize("count", [63, 64])
def test_decode_benchmark_requires_completed_output_budget(monkeypatch, tmp_path, count):
    monkeypatch.setattr(decode_benchmark, "request", lambda *args: {"response": {"tokens": [1, 2]}})
    monkeypatch.setattr(
        decode_benchmark, "stream_measure", lambda *args, **kwargs: dict(usage=dict(completion_tokens=count))
    )
    output = tmp_path / "result.json"
    if count == 64:
        assert decode_benchmark.run("http://localhost", output)["passed"]
    else:
        with pytest.raises(AssertionError, match="output budget"):
            decode_benchmark.run("http://localhost", output)
        assert json.loads(output.read_text())["passed"] is False


@pytest.mark.parametrize("kwargs", [dict(max_tokens=True), dict(max_tokens=0), dict(ignore_eos=1)])
def test_decode_controls_reject_ambiguous_values(kwargs):
    with pytest.raises(ValueError, match="output budget"):
        prefill_benchmark.stream_measure("http://localhost", [1], **kwargs)
