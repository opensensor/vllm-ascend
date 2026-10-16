# SPDX-License-Identifier: Apache-2.0
import io
import json

import pytest

from tools.qwen4exp import benchmark_tp6_stability as benchmark


@pytest.mark.parametrize("failure", [None, "error", "missing_done", "missing_usage", "empty"])
def test_stream_completion_requires_content_usage_and_done(monkeypatch, failure):
    events = []
    if failure == "error":
        events.append(dict(error="engine failed"))
    if failure != "empty":
        events.append(dict(choices=[dict(delta=dict(content="42"))]))
    if failure != "missing_usage":
        events.append(dict(choices=[], usage=dict(prompt_tokens=12, completion_tokens=1)))
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    if failure != "missing_done":
        body += b"data: [DONE]\n\n"
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(body))
    if failure:
        with pytest.raises(RuntimeError):
            benchmark.completion("http://unused", "qfn", "19+23", 32, 1)
    else:
        result = benchmark.completion("http://unused", "qfn", "19+23", 32, 1)
        assert result["text"] == "42" and result["usage"]["completion_tokens"] == 1
