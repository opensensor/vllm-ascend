# SPDX-License-Identifier: Apache-2.0
"""Decode observation retains cold prefill geometry and restores observer state."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.expert_actor import ExpertCoordinator
from tools.sidecard.expert_profile import ProfileCoordinator
from tools.sidecard.expert_profile import run as profile_run


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("decode_only", [False, True])
def test_prefill_observers_restore_state_and_keep_admitted_window(monkeypatch, fail, decode_only):
    engine = object.__new__(ProfileCoordinator)
    engine.enabled, engine.decode_only = True, decode_only
    engine.prefill_window_size, engine.workload_logits = 1024, None
    seen = []
    result = object()

    def prefill(owner, tokens, **kwargs):
        seen.append((owner.enabled, owner.prefill_window_size, tokens, kwargs))
        if fail:
            raise RuntimeError("inference failed")
        return result

    monkeypatch.setattr(ExpertCoordinator, "prefill", prefill)
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            engine.prefill([1, 2], overlap=False)
    else:
        assert engine.prefill([1, 2], overlap=False) is result
    assert seen == [(not decode_only, 1024, [1, 2], {"overlap": False})]
    assert engine.enabled


def test_optional_vocabulary_observer_hashes_all_logits_and_can_be_disabled():
    engine = object.__new__(ProfileCoordinator)
    logits = torch.tensor([0.125, -100.0, 5.0])
    engine.workload_logits = []
    engine.observe_logits(logits)
    assert engine.workload_logits == [hashlib.sha256(logits.numpy().tobytes()).hexdigest()]
    engine.workload_logits = None
    engine.observe_logits(object())


@pytest.mark.parametrize("error", ["prefix_cache", "missing_reference", "authentication", "failed_reference"])
def test_decode_profile_rejects_unqualified_inputs_before_model_allocation(tmp_path, monkeypatch, error):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object()))
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps(dict(expert_transport="direct-dma", prompt_prefix_cache=error == "prefix_cache")))
    reference = tmp_path / "benchmark.json"
    reference.write_text(json.dumps(dict(passed=error != "failed_reference")))
    signature = hashlib.sha256(reference.read_bytes()).hexdigest()
    if error == "authentication":
        signature = "0" * 64
    if error == "missing_reference":
        signature = None
    with pytest.raises(ValueError):
        profile_run(
            config,
            tmp_path / "missing-transport-reference.json",
            decode_only=True,
            chat_request=tmp_path / "private-request.json",
            benchmark_reference=reference,
            benchmark_sha256=signature,
        )
