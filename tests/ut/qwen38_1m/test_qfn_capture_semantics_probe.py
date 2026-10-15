# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU model of record-only and record-plus-execute graph semantics."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.qwen4exp.resident_candidates import qfn_capture_semantics_probe as probe


class State:
    def __init__(self, runtime):
        self.runtime = runtime
        self.value = 0.0

    def add_(self, amount):
        capture = self.runtime.active
        if capture is not None and capture._capturing:
            capture.graph.callbacks.append(lambda: self.apply(amount))
            if self.runtime.executes_capture:
                self.apply(amount)
        else:
            self.apply(amount)
        return self

    def apply(self, amount):
        self.value += amount

    def cpu(self):
        return self

    def item(self):
        return self.value


class Graph:
    def __init__(self, runtime):
        self.runtime = runtime
        self.callbacks = []
        self.resets = 0
        runtime.graphs.append(self)

    def replay(self):
        if self.runtime.replay_error:
            raise ValueError("injected replay failure")
        for callback in self.callbacks:
            callback()

    def reset(self):
        self.resets += 1
        if self.runtime.reset_error:
            raise ValueError("injected reset failure")


class Runtime:
    def __init__(self, executes_capture, *, replay_error=False, reset_error=False):
        self.active = None
        self.executes_capture = executes_capture
        self.replay_error = replay_error
        self.reset_error = reset_error
        self.graphs = []
        self.stream = SimpleNamespace(synchronize=Mock(), wait_stream=Mock())
        self.npu = SimpleNamespace(
            Stream=lambda: self.stream,
            current_stream=lambda: "caller-stream",
            stream=lambda stream: nullcontext(),
            current_device=lambda: 2,
        )
        self.float32 = "float32"
        runtime = self

        class Capture:
            def __init__(self):
                self.segments = []
                self._current_graph = None
                self._capturing = False
                self.num_graphs = 0
                self.num_eager_breaks = 0

            @classmethod
            def current(cls):
                return runtime.active

            def begin(self):
                self.graph = Graph(runtime)
                self._current_graph = self.graph
                self._capturing = True

            def end(self):
                self.segments.append(self.graph.replay)
                self.num_graphs += 1
                self._current_graph = None
                self._capturing = False

            def __enter__(self):
                runtime.active = self
                self.begin()
                return self

            def __exit__(self, *args):
                self.end()
                runtime.active = None

            def add_eager(self, callback):
                self.end()
                callback()
                self.segments.append(callback)
                self.num_eager_breaks += 1
                self.begin()

            def replay(self):
                for segment in self.segments:
                    segment()

        self.capture_type = Capture

    def inference_mode(self):
        return nullcontext()

    def zeros(self, size, *, dtype, device):
        assert size == 1 and dtype == "float32" and device == "npu:2"
        return State(self)

    def zeros_like(self, state):
        return State(self)


@pytest.mark.parametrize("executes_capture", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_observes_both_capture_semantics_without_assuming_either(executes_capture, mixed):
    runtime = Runtime(executes_capture)
    report = probe.observe_capture(runtime, runtime.capture_type, mixed=mixed)
    expected_capture = (111 if mixed else 1) if executes_capture else (10 if mixed else 0)
    assert report["passed"] is True
    assert report["state_after_capture"] == expected_capture
    assert report["state_after_one_replay"] == expected_capture + (111 if mixed else 1)
    assert report["eager_calls_after_capture"] == int(mixed)
    assert report["eager_calls_after_replay"] == 2 * int(mixed)
    if mixed:
        observations = report["eager_observations"]
        assert observations[0]["before_eager"] == int(executes_capture)
        assert observations[0]["after_eager"] == int(executes_capture) + 10
        assert observations[1]["before_eager"] == expected_capture + 1
        assert observations[1]["after_eager"] == expected_capture + 11
    else:
        assert report["eager_observations"] == []
    assert report["reset_graphs"] == (2 if mixed else 1)
    assert all(graph.resets == 1 for graph in runtime.graphs)
    assert runtime.active is None
    runtime.stream.wait_stream.assert_called_once_with("caller-stream")


def test_failed_replay_still_resets_only_owned_graphs():
    runtime = Runtime(False, replay_error=True)
    report = probe.observe_capture(runtime, runtime.capture_type, mixed=True)
    assert report["passed"] is False
    assert "injected replay failure" in report["error"]
    assert report["reset_graphs"] == 2
    assert all(graph.resets == 1 for graph in runtime.graphs)


def test_cleanup_failure_is_reported_per_rank():
    runtime = Runtime(False, reset_error=True)
    report = probe.observe_capture(runtime, runtime.capture_type, mixed=True)
    assert report["passed"] is False
    assert len(report["reset_errors"]) == 2
    assert all(graph.resets == 1 for graph in runtime.graphs)


def test_failed_context_entry_closes_its_thread_local_capture_and_graph():
    runtime = Runtime(False)
    base = runtime.capture_type

    class FailedEntry(base):
        def __enter__(self):
            super().__enter__()
            raise RuntimeError("entry failed after starting capture")

    report = probe.observe_capture(runtime, FailedEntry, mixed=True)
    assert report["passed"] is False
    assert "entry failed" in report["error"]
    assert runtime.active is None
    assert report["reset_graphs"] == 1
    assert runtime.graphs[0].resets == 1


def test_existing_capture_is_never_touched():
    runtime = Runtime(False)
    external = object()
    runtime.active = external
    report = probe.observe_capture(runtime, runtime.capture_type, mixed=True)
    assert report["passed"] is False
    assert runtime.active is external
    assert not runtime.graphs


def test_rpc_retains_normal_profile_stop_with_output_argument():
    original = Mock(return_value="profile stopped")
    run = Mock(side_effect=AssertionError("unexpected probe"))
    worker = SimpleNamespace(rank=3)
    rpc = probe.make_rpc(original, run)
    assert rpc(worker, "/tmp/profile") == "profile stopped"
    original.assert_called_once_with(worker, "/tmp/profile")
    run.assert_not_called()


def test_rpc_without_output_runs_gate_and_keeps_rank():
    run = Mock(return_value={"passed": True})
    rpc = probe.make_rpc(Mock(), run)
    assert rpc(SimpleNamespace(rank=2)) == {"rank": 2, "capture_semantics": {"passed": True}}
    run.assert_called_once_with()


@pytest.mark.parametrize("error", [ValueError("bad"), SystemExit("stop")])
def test_rpc_contains_all_probe_failures_per_rank(error):
    rpc = probe.make_rpc(Mock(), Mock(side_effect=error))
    result = rpc(SimpleNamespace(rank=1))
    assert result["rank"] == 1
    assert result["capture_semantics"]["passed"] is False
    assert type(error).__name__ in result["capture_semantics"]["error"]


def test_active_cpu_profiler_rejects_cold_probe():
    run = Mock()
    result = probe.make_rpc(Mock(), run)(SimpleNamespace(rank=0, _qfn_cpu_profile=object()))
    assert result["capture_semantics"]["passed"] is False
    run.assert_not_called()


def test_replacement_targets_existing_profile_stop_only(monkeypatch):
    module = ModuleType("vllm_ascend._310p.worker_310p")
    stop = Mock()
    inventory = Mock()
    module.NPUWorker310 = SimpleNamespace(qfn_cpu_profile_stop=stop, qfn_graph_inventory=inventory)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    replacements = probe.replacements()
    assert list(replacements) == ["vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"]
    assert module.NPUWorker310.qfn_graph_inventory is inventory
    assert module.NPUWorker310.qfn_cpu_profile_stop is stop
