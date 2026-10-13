# SPDX-License-Identifier: Apache-2.0
"""Matched cold compute measurements reject numerical changes and cache use."""

import json
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.cold_prefill_benchmark import run


@pytest.mark.parametrize("change", [False, True])
def test_cold_benchmark_resets_every_case_and_records_all_decode_logits(tmp_path, change):
    calls = []
    engine = SimpleNamespace(prefill_chunk_size=32, prefill_window_size=64, fingerprints=["original"], max_context=4096)

    def reset(**kwargs):
        calls.append((engine.prefill_window_size, kwargs))
        engine.position = 0

    def forward(tokens):
        engine.position += len(tokens)
        altered = int(change and engine.prefill_window_size == 64 and engine.position > 65)
        return torch.tensor([engine.position * 0.01, 1.0 + altered])

    engine.reset, engine.forward, engine.prefill = reset, forward, forward
    output = tmp_path / "benchmark.json"
    if change:
        with pytest.raises(AssertionError, match="full-vocabulary"):
            run(engine, [("full-code", list(range(64)))], output=output)
    else:
        result = run(engine, [("full-code", list(range(64)))], output=output)
        assert result["passed"] and result["cached_prompt_tokens"] == 0
    report = json.loads(output.read_text())
    rows = report["measurements"]
    assert calls == [(32, {"context_limit": 128}), (64, {"context_limit": 128})]
    assert rows[0]["prompt_ids_sha256"] == rows[1]["prompt_ids_sha256"]
    assert len(rows[1]["logits_sha256"]) == 64
    assert rows[1]["full_vocabulary_bitwise_equal"] is not change
    assert engine.prefill_window_size == 64 and engine.fingerprints == ["original"]
    if change:
        assert not report["passed"] and report["error"]["type"] == "AssertionError"


def test_cold_benchmark_rejects_insufficient_admission_and_restores_configuration(tmp_path):
    engine = SimpleNamespace(prefill_chunk_size=32, prefill_window_size=64, fingerprints=None, max_context=64)
    with pytest.raises(ValueError, match="capacity"):
        run(engine, [("full-code", [1])], output=tmp_path / "benchmark.json")
    assert engine.prefill_window_size == 64
    engine.prefill_window_size = 32
    with pytest.raises(ValueError, match="requires"):
        run(engine, [("full-code", [1])], output=tmp_path / "benchmark.json")


@pytest.mark.parametrize("windows", [(64, 128), (32, 64, 128)])
def test_larger_benchmark_uses_first_window_as_teacher_and_checks_each_candidate(tmp_path, windows):
    seen = []
    engine = SimpleNamespace(prefill_chunk_size=32, prefill_window_size=128, fingerprints=None, max_context=512)
    engine.reset = lambda **kwargs: seen.append(engine.prefill_window_size)
    engine.prefill = engine.forward = lambda tokens: torch.tensor([0.0, 1.0])
    report = run(engine, [("full-code", list(range(128)))], output=tmp_path / "benchmark.json", windows=windows)
    assert report["passed"] and seen == list(windows)
    assert all(row["full_vocabulary_bitwise_equal"] for row in report["measurements"])
    assert engine.prefill_window_size == 128


@pytest.mark.parametrize("windows", [(128,), (64, 64), (64, 256), (True, 64), (64, 128)])
def test_benchmark_rejects_unadmitted_windows_before_reset(tmp_path, windows):
    engine = SimpleNamespace(prefill_chunk_size=32, prefill_window_size=64, fingerprints=None)
    with pytest.raises(ValueError, match="requires"):
        run(engine, [("full-code", [1])], output=tmp_path / "benchmark.json", windows=windows)
