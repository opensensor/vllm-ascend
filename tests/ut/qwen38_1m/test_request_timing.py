# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.observability.request_timing import AscendRequestTimingLogger


def _request(generated=5):
    return SimpleNamespace(
        num_prompt_tokens=100,
        num_cached_tokens=80,
        num_generation_tokens=generated,
        prefill_time=2.0,
        decode_time=1.0,
        request_id="timing-test",
        finish_reason="stop",
        queued_time=0.1,
        e2e_latency=3.1,
    )


def test_request_timing_excludes_cached_prompt_and_first_token_gap(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LOG_REQUEST_TIMINGS", "1")
    timing = AscendRequestTimingLogger(None, engine_index=2)
    with patch("vllm_ascend.observability.request_timing.logger.info") as log:
        timing.record(None, SimpleNamespace(finished_requests=[_request()]))
    message, *args = log.call_args.args
    rendered = message % tuple(args)
    assert "Engine 002" in rendered
    assert "prompt 20 computed + 80 cached in 2000.0 ms (100.0 ms/tok, 10.0 effective tok/s)" in rendered
    assert "decode 5 tokens, 4 gaps in 1000.0 ms (250.0 ms/tok, 4.0 tok/s)" in rendered


def test_request_timing_is_opt_in_and_ignores_empty_iterations(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_LOG_REQUEST_TIMINGS", raising=False)
    timing = AscendRequestTimingLogger(None)
    with patch("vllm_ascend.observability.request_timing.logger.info") as log:
        timing.record(None, SimpleNamespace(finished_requests=[_request()]))
        timing.record(None, None)
    log.assert_not_called()


def test_request_timing_single_token_does_not_invent_decode_rate(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LOG_REQUEST_TIMINGS", "1")
    timing = AscendRequestTimingLogger(None)
    with patch("vllm_ascend.observability.request_timing.logger.info") as log:
        timing.record(None, SimpleNamespace(finished_requests=[_request(generated=1)]))
    message, *args = log.call_args.args
    assert "decode 1 tokens, 0 gaps in n/a ms (n/a ms/tok, n/a tok/s)" in message % tuple(args)
