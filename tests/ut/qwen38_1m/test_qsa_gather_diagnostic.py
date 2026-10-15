# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools.qwen4exp import qsa_gather_diagnostic as diagnostic


def environment(monkeypatch):
    events = []

    class Wrapper:
        def __init__(self, config, name):
            self._active = False
            self.is_running = False
            self.config = config
            self.name = name

        def _start(self):
            events.append("start")

        def _stop(self):
            events.append("stop")

        def start(self):
            self._start()
            self._active = self.is_running = True

        def stop(self):
            self._stop()
            self._active = self.is_running = False

    worker = SimpleNamespace(rank=2)
    state = dict(resources=dict(gather=object()))
    torch = SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: events.append("sync")),
        profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
    )

    def timing(resource, device, **kwargs):
        assert resource is state["resources"]["gather"] and device == "npu:2"
        assert kwargs["record_function"] is torch.profiler.record_function
        assert worker.profiler is state["profile_session"].wrapper
        events.append("timing")
        return dict(passed=True, records=[dict(variant=8)])

    monkeypatch.setattr(diagnostic.qsa_gather_staging, "timing", timing)
    deps = lambda: (lambda **kwargs: SimpleNamespace(**kwargs), Wrapper)
    return worker, state, torch, deps, events


def run(worker, state, torch, deps, tmp_path):
    return diagnostic.profile(
        worker,
        state,
        "npu:2",
        rows=[64],
        repeats=2,
        output=str(tmp_path / "trace"),
        dependency_factory=deps,
        torch_module=torch,
    )


def test_balanced_profile_one_shot_preserves_session_receipt(monkeypatch, tmp_path):
    worker, state, torch, deps, events = environment(monkeypatch)
    receipt = run(worker, state, torch, deps, tmp_path)
    assert events == ["start", "timing", "sync", "stop"]
    assert receipt["passed"] and receipt["backend_starts"] == receipt["backend_stops"] == 1
    assert receipt["output_dir"] == str(tmp_path / "trace/rank2")
    assert worker.profiler is worker._qfn_native_profile_session is None
    assert state["profile_session"].wrapper is not None
    with pytest.raises(RuntimeError, match="one-shot"):
        run(worker, state, torch, deps, tmp_path)
    assert events == ["start", "timing", "sync", "stop"]


@pytest.mark.parametrize("attribute", ["profiler", "_qfn_cpu_profile", "_qfn_native_profile_session"])
def test_rejects_any_existing_profiler_before_creating_session(monkeypatch, tmp_path, attribute):
    worker, state, torch, deps, events = environment(monkeypatch)
    setattr(worker, attribute, object())
    with pytest.raises(RuntimeError, match="another profiler"):
        run(worker, state, torch, deps, tmp_path)
    assert not events and "profile_session" not in state


def test_native_failure_retains_active_profiler_without_automatic_stop(monkeypatch, tmp_path):
    worker, state, torch, deps, events = environment(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("native launch uncertain")

    monkeypatch.setattr(diagnostic.qsa_gather_staging, "timing", fail)
    with pytest.raises(RuntimeError, match="native launch uncertain"):
        run(worker, state, torch, deps, tmp_path)
    session = state["profile_session"]
    assert worker.profiler is session.wrapper and worker._qfn_native_profile_session is session
    assert session.starts == 1 and session.stops == 0 and events == ["start"]


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_backend_error_retains_owner_and_records_actual_phase(monkeypatch, tmp_path, operation):
    worker, state, torch, deps, events = environment(monkeypatch)
    config, wrapper = deps()

    def fail(self):
        raise RuntimeError("backend error")

    setattr(wrapper, "_" + operation, fail)
    with pytest.raises(RuntimeError, match="backend error"):
        run(worker, state, torch, lambda: (config, wrapper), tmp_path)
    session = state["profile_session"]
    assert session.error == "backend " + operation + ": RuntimeError: backend error"
    assert worker.profiler is session.wrapper and worker._qfn_native_profile_session is session
    assert session.stops == 0


def test_constructor_failure_publishes_uncertain_session_before_creation(monkeypatch, tmp_path):
    worker, state, torch, deps, events = environment(monkeypatch)
    config, _ = deps()

    def fail(*args):
        raise RuntimeError("construct failed")

    with pytest.raises(RuntimeError, match="construct failed"):
        run(worker, state, torch, lambda: (config, fail), tmp_path)
    assert worker._qfn_native_profile_session is state["profile_session"]
    assert state["profile_session"].wrapper is None and not events
