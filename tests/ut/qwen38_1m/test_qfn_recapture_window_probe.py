# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-window CPU tests using a private fake device and real window code."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.qwen38_1m.test_qfn_capture_semantics_probe import Runtime, State
from tools.qwen4exp.resident_candidates import qfn_recapture_window_probe as probe
from tools.qwen4exp.resident_candidates import qsa_graph_dispatch as qsa


class WindowState(State):
    def clone(self):
        result = WindowState(self.runtime)
        result.value = self.value
        return result

    def __mul__(self, factor):
        result = self.clone()
        result.value *= factor
        return result


class WindowRuntime(Runtime):
    def __init__(self, executes_capture, *, replay_error=False):
        super().__init__(executes_capture, replay_error=replay_error)
        self.timeline = []
        runtime = self
        streams = []
        fences = []

        class Stream:
            def __init__(self):
                self.index = len(streams)
                streams.append(self)

            def synchronize(self):
                runtime.timeline.append(("sync", self.index))

            def wait_event(self, event):
                runtime.timeline.append(("wait", self.index, event.index))

        class Event:
            def __init__(self):
                self.index = len(fences)
                fences.append(self)

            def record(self, stream):
                runtime.timeline.append(("record", stream.index, self.index))

        default = Stream()
        current = [default]

        @contextmanager
        def use_stream(stream):
            prior = current[0]
            current[0] = stream
            runtime.timeline.append(("enter", stream.index))
            try:
                yield
            finally:
                current[0] = prior
                runtime.timeline.append(("exit", stream.index))

        self.npu = SimpleNamespace(
            Stream=Stream,
            Event=Event,
            current_stream=lambda: current[0],
            default_stream=lambda: default,
            stream=use_stream,
            current_device=lambda: 2,
        )
        self.streams = streams
        self.fences = fences

    def zeros(self, size, *, dtype, device):
        assert size == 1 and dtype == "float32" and device == "npu:2"
        return WindowState(self)

    def zeros_like(self, other):
        return WindowState(self)


@pytest.mark.parametrize("previous_monitor", [False, True])
@pytest.mark.parametrize("executes_capture", [False, True])
def test_exact_window_default_stream_producer_consumer_and_resource_reuse(
    monkeypatch, previous_monitor, executes_capture
):
    from vllm.compilation import monitor

    runtime = WindowRuntime(executes_capture)
    monkeypatch.setattr(qsa, "torch", runtime)
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", previous_monitor)
    dispatcher = qsa.PrefixDispatcher({})
    for mixed in (False, True):
        report = probe.observe_window(
            runtime, runtime.capture_type, monitor, dispatcher, qsa.serialized_recapture_window, mixed=mixed
        )
        expected_capture = 7 + ((111 if mixed else 1) if executes_capture else (10 if mixed else 0))
        assert report["passed"]
        assert report["producer_value"] == 7
        assert report["state_after_capture"] == expected_capture
        assert report["state_after_one_replay"] == expected_capture + (111 if mixed else 1)
        assert report["consumer_value"] == 2 * report["state_after_one_replay"]
        assert report["caller_stream_restored"] and report["monitor_restored"]
        assert report["resources_retained"] and report["resources_reused"]
        assert report["reset_graphs"] == (2 if mixed else 1)
    fences = [event for event in runtime.timeline if event[0] in ("record", "wait")]
    assert fences == [("record", 0, 0), ("wait", 1, 0), ("record", 1, 1), ("wait", 0, 1)] * 2
    assert len(runtime.streams) == 2 and len(runtime.fences) == 2
    assert all(graph.resets == 1 for graph in runtime.graphs)
    assert not qsa.RESIDENT_RECAPTURE_WINDOW_QUALIFIED


def test_failed_replay_restores_monitor_stream_and_resets_owned_graphs(monkeypatch):
    from vllm.compilation import monitor

    runtime = WindowRuntime(False, replay_error=True)
    monkeypatch.setattr(qsa, "torch", runtime)
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", False)
    dispatcher = qsa.PrefixDispatcher({})
    report = probe.observe_window(
        runtime, runtime.capture_type, monitor, dispatcher, qsa.serialized_recapture_window, mixed=True
    )
    assert not report["passed"]
    assert "replay failure" in report["error"]
    assert report["caller_stream_restored"] and report["monitor_restored"]
    assert report["reset_graphs"] == 2
    assert dispatcher.failed and dispatcher.recapture_resources is not None
    fences = [event for event in runtime.timeline if event[0] in ("record", "wait")]
    assert fences == [("record", 0, 0), ("wait", 1, 0)]


def test_nondefault_caller_rejected_before_capture_or_window(monkeypatch):
    from vllm.compilation import monitor

    runtime = WindowRuntime(False)
    dispatcher = qsa.PrefixDispatcher({})
    window = Mock(side_effect=AssertionError("must not open window"))
    with runtime.npu.stream(runtime.npu.Stream()):
        report = probe.observe_window(runtime, runtime.capture_type, monitor, dispatcher, window, mixed=True)
    assert not report["passed"] and "default stream" in report["error"]
    window.assert_not_called()
    assert not runtime.graphs


def test_replacement_installs_only_cold_rpc_without_qsa_factory(monkeypatch):
    module = ModuleType("vllm_ascend._310p.worker_310p")
    module.NPUWorker310 = SimpleNamespace(qfn_cpu_profile_stop=Mock())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(qsa, "replacements", Mock(side_effect=AssertionError("QSA factory must remain blocked")))
    assert list(probe.replacements()) == ["vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"]
    qsa.replacements.assert_not_called()
    assert not qsa.RESIDENT_RECAPTURE_WINDOW_QUALIFIED


def test_stream_identity_accepts_distinct_equal_python_wrappers(monkeypatch):
    from vllm.compilation import monitor

    runtime = WindowRuntime(False)
    native_current = runtime.npu.current_stream
    wrappers = []

    class StreamView:
        def __init__(self, native):
            self.native = native
            wrappers.append(self)

        def __eq__(self, other):
            return self.native is getattr(other, "native", other)

        def __getattr__(self, name):
            return getattr(self.native, name)

    runtime.npu.current_stream = lambda: StreamView(native_current())
    monkeypatch.setattr(qsa, "torch", runtime)
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", False)
    dispatcher = qsa.PrefixDispatcher({})
    for mixed in (False, True):
        report = probe.observe_window(
            runtime,
            runtime.capture_type,
            monitor,
            dispatcher,
            qsa.serialized_recapture_window,
            mixed=mixed,
        )
        assert report["passed"]
        assert report["resources_retained"] and report["resources_reused"]
    retained = dispatcher.recapture_resources[0]
    equivalent = [wrapper for wrapper in wrappers if wrapper == retained]
    assert len(equivalent) > 1
    assert all(wrapper is not retained for wrapper in equivalent)
    assert equivalent[0] is not equivalent[1]
    assert all(graph.resets == 1 for graph in runtime.graphs)
