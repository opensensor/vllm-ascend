# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.test_qfn_early_tp_draft import SOURCE, setup_runner
from tools.qwen4exp import output_observer
from tools.qwen4exp.output_observer import OutputObserver
from tools.qwen4exp.resident_candidates import qfn_output_observer as candidate
from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC


def test_exact_pinned_sample_submits_draft_before_observation_join():
    gate = setup_runner()

    def start(*args):
        gate.events.append("copy_and_observe")
        return True

    observer = SimpleNamespace(start=start)
    forward = candidate.build_forward(gate.original, observer, source=SOURCE)
    forward(gate.runner, None)
    assert gate.events == ["copy_and_observe", "propose", "copy_draft", "bookkeeping", "finalize"]
    assert gate.input_batch.num_tokens_no_spec == [11]


def test_unsupported_observation_keeps_original_bookkeeping_order():
    gate = setup_runner()
    forward = candidate.build_forward(gate.original, SimpleNamespace(start=lambda *args: False), source=SOURCE)
    forward(gate.runner, None)
    assert gate.events == ["bookkeeping", "propose", "copy_draft", "finalize"]


def test_source_drift_is_rejected():
    gate = setup_runner()
    with pytest.raises(RuntimeError, match="exact qualified"):
        candidate.build_forward(gate.original, None, source=SOURCE + "\n")


def owner_fixture():
    owner = OutputObserver.__new__(OutputObserver)
    owner.failed = False
    owner.future = Future()
    owner.observed = owner.completed_before_join = 0
    return owner


def test_observation_lifetime_is_retired_once():
    owner = owner_fixture()
    output = object()
    owner.future.set_result(output)
    assert owner.finish() is output
    assert owner.future is None and not owner.failed
    assert owner.observed == owner.completed_before_join == 1
    with pytest.raises(RuntimeError, match="no output observation"):
        owner.finish()


def test_failed_observation_retains_future_and_cannot_be_resubmitted():
    owner = owner_fixture()
    future = owner.future
    future.set_exception(ValueError("bad output"))
    with pytest.raises(ValueError, match="bad output"):
        owner.finish()
    assert owner.failed and owner.future is future
    with pytest.raises(RuntimeError, match="unretired or failed"):
        owner.start(None, None, None, None, None, None, None, False)


def test_observer_wrapper_does_not_repeat_device_commit_or_host_bookkeeping(monkeypatch):
    monkeypatch.setattr(candidate, "build_forward", lambda *args: lambda: None)
    runner = SimpleNamespace(sample_tokens=lambda: None)
    owner = owner_fixture()
    owner.runner = runner
    owner.future.set_result("observed")
    calls = []
    base = {RUNNER + "._bookkeeping_sync": lambda *args: calls.append("fallback"), RPC: lambda *args: "old"}
    wrappers = candidate.make_replacements(base, owner, runner)
    assert wrappers[RUNNER + "._bookkeeping_sync"](runner) == "observed"
    assert calls == []
    wrappers[RUNNER + "._bookkeeping_sync"](runner)
    assert calls == ["fallback"]
    assert wrappers[RPC](None, True) == "old"


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_copy_stream_waits_on_producer_not_on_itself(monkeypatch, dtype):
    @dataclass
    class Sampler:
        sampled_token_ids: object
        logprobs_tensors: object = None

    calls = []
    producer = object()
    current = [producer]
    copy_stream = SimpleNamespace(wait_stream=lambda stream: calls.append(("wait", stream)))

    @contextmanager
    def stream_context(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda device: current[0], stream=stream_context), raising=False
    )
    monkeypatch.setattr(output_observer, "eligible", lambda *args: True)
    owner = OutputObserver.__new__(OutputObserver)
    owner.enabled, owner.failed, owner.future = True, False, None
    owner.runner = SimpleNamespace(device="cpu")
    owner.transition = SimpleNamespace(enabled=True, verify=False, begin=lambda *args: True)
    owner.bookkeeping = lambda *args: None
    owner.host_samples = {dtype: torch.empty((4, 3), dtype=dtype)}
    owner.copy_stream = copy_stream
    owner.copy_done = SimpleNamespace(record=lambda: calls.append("record"))
    owner.executor = SimpleNamespace(submit=lambda *args: Future())
    samples = Sampler(torch.tensor([[7, 8, -1]], dtype=dtype))
    assert owner.start(owner.runner, samples, None, None, None, None, None, True)
    assert calls == [("wait", producer), "record"]
    assert owner.host_samples[dtype][0].tolist() == [7, 8, -1]
