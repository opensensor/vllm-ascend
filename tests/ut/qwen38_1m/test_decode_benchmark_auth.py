# SPDX-License-Identifier: Apache-2.0
import io

import pytest

from tools.qwen38_decode_study import benchmark


def test_metrics_auth_uses_private_token_file(tmp_path, monkeypatch):
    token = tmp_path / "control.token"
    token.write_text("ab" * 32)
    token.chmod(0o600)
    requests = []

    def open_url(request, timeout):
        requests.append(request)
        assert timeout == 10
        return io.BytesIO(b'vllm:spec_decode_num_draft_tokens_total{model="qfn"} 12\n')

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", open_url)
    assert benchmark.metrics("http://127.0.0.1:8002", token) == {"vllm:spec_decode_num_draft_tokens_total": 12}
    assert requests[-1].get_header("X-qfn-resident-token") == "ab" * 32
    benchmark.metrics("http://127.0.0.1:8002")
    assert requests[-1].get_header("X-qfn-resident-token") is None
    with pytest.raises(ValueError, match="loopback"):
        benchmark.metrics("http://example.com:8002", token)
    assert len(requests) == 2
    token.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        benchmark.metrics("http://127.0.0.1:8002", token)
