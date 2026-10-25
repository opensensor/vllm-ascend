# SPDX-License-Identifier: Apache-2.0
"""Check explicit-budget warmup ordering and the model guard."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.minimax_memory import W4_LAYOUT, warmup_minimax_before_cache


def test_model_is_warmed_before_cache_release(monkeypatch):
    events = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: events.append("sync"), empty_cache=lambda: events.append("release")),
        raising=False,
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(ascend_minimax_m3_w4=W4_LAYOUT)),
        device="cpu",
        model_runner=SimpleNamespace(profile_run=lambda: events.append("profile")),
    )
    assert warmup_minimax_before_cache(worker)
    assert events == ["profile", "sync", "release"]


def test_other_models_keep_the_existing_explicit_budget_fast_path():
    assert not warmup_minimax_before_cache(SimpleNamespace())


def test_warmup_errors_are_not_hidden(monkeypatch):
    def fail():
        raise RuntimeError("communication failed")

    worker = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(ascend_minimax_m3_w4=W4_LAYOUT)),
        model_runner=SimpleNamespace(profile_run=fail),
    )
    with pytest.raises(RuntimeError, match="communication failed"):
        warmup_minimax_before_cache(worker)
