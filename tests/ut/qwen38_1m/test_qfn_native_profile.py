# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded profiling lifecycle without torch_npu or an initialized worker."""

import json
import sys
from contextlib import suppress
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.qwen4exp.resident_candidates import qfn_native_profile as candidate


class FakeWrapper:
    """Model the qualified wrapper's suppressed errors and pre-forward step."""

    def __init__(self, config, trace_name):
        self.config, self.trace_name = config, trace_name
        self._active = self.is_running = False
        self.steps = self.recorded = 0
        self.fail_start = self.fail_stop = False

    def _start(self):
        if self.fail_start:
            raise RuntimeError("start failure")

    def _stop(self):
        if self.fail_stop:
            raise RuntimeError("stop failure")

    def call_start(self):
        try:
            self._start()
            self.is_running = True
        except RuntimeError:
            pass

    def start(self):
        self._active = True
        if self.config.delay_iterations == 0:
            self.call_start()

    def step(self):
        if not self._active:
            return
        self.steps += 1
        if not self.is_running and self.steps == self.config.delay_iterations:
            self.call_start()
        if self.is_running:
            self.recorded += 1
        if self.recorded > self.config.max_iterations:
            self.stop()

    def stop(self):
        if not self._active:
            return
        self._active = False
        if self.is_running:
            with suppress(RuntimeError):
                self._stop()
            self.is_running = False


def setup(tmp_path, *, delay=0, maximum=8, fail_start=False):
    worker = SimpleNamespace(rank=2, profiler=None, profiler_config=object())
    original_config = worker.profiler_config
    cpu_start, cpu_stop = Mock(return_value={"cpu": "start"}), Mock(return_value={"cpu": "stop"})

    def factory(config, name):
        wrapper = FakeWrapper(config, name)
        wrapper.fail_start = fail_start
        return wrapper

    start, stop = candidate.make_rpcs(cpu_start, cpu_stop, dependency_factory=lambda: (SimpleNamespace, factory))
    payload = json.dumps(dict(output_dir=str(tmp_path), tag="decode", delay=delay, max=maximum))
    return SimpleNamespace(**locals())


def test_start_stop_preserves_config_and_restores_profiler(tmp_path):
    gate = setup(tmp_path)
    result = gate.start(gate.worker, gate.payload)
    assert result["running"] and result["armed"] and result["backend_starts"] == 1
    wrapper = gate.worker.profiler
    assert wrapper.trace_name == "decode_rank2"
    assert wrapper.config.ignore_frontend and not wrapper.config.torch_profiler_with_stack
    assert not wrapper.config.torch_profiler_with_memory
    assert gate.worker.profiler_config is gate.original_config
    result = gate.stop(gate.worker)
    assert result["ownership_released"] and result["recorded"]
    assert result["backend_starts"] == result["backend_stops"] == 1
    assert gate.worker.profiler is gate.worker._qfn_native_profile_session is None
    assert gate.worker.profiler_config is gate.original_config


def test_delayed_start_and_automatic_iteration_cap(tmp_path):
    gate = setup(tmp_path, delay=2, maximum=3)
    result = gate.start(gate.worker, gate.payload)
    assert result["armed"] and not result["running"]
    wrapper = gate.worker.profiler
    wrapper.step()
    assert not wrapper.is_running
    wrapper.step()
    assert wrapper.is_running
    for _ in range(3):
        wrapper.step()
    assert not wrapper.is_running and not wrapper._active
    result = gate.stop(gate.worker)
    assert result["backend_starts"] == result["backend_stops"] == 1
    assert result["ownership_released"]


def test_cancel_before_delay_reports_no_recording(tmp_path):
    gate = setup(tmp_path, delay=8)
    gate.start(gate.worker, gate.payload)
    result = gate.stop(gate.worker)
    assert result["ownership_released"] and not result["recorded"]


@pytest.mark.parametrize("when", ["start", "delayed_start", "stop", "automatic_stop"])
def test_backend_errors_survive_wrapper_suppression_and_retain_ownership(tmp_path, when):
    gate = setup(tmp_path, delay=1 if when == "delayed_start" else 0, fail_start=when == "start")
    started = gate.start(gate.worker, gate.payload)
    wrapper = gate.worker.profiler
    if when == "start":
        assert "start failure" in started["error"]
    elif when == "delayed_start":
        wrapper.fail_start = True
        wrapper.step()
    else:
        wrapper.fail_stop = True
        if when == "automatic_stop":
            for _ in range(9):
                wrapper.step()
    result = gate.stop(gate.worker)
    assert "failure" in result["error"]
    assert gate.worker.profiler is wrapper
    assert gate.worker._qfn_native_profile_session is not None
    assert "ownership_released" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("delay", -1),
        ("delay", 129),
        ("delay", True),
        ("max", 0),
        ("max", 65),
        ("tag", "../x"),
        ("output_dir", "relative"),
    ],
)
def test_invalid_configuration_rejected_before_install(tmp_path, field, value):
    gate = setup(tmp_path)
    config = json.loads(gate.payload)
    config[field] = value
    result = gate.start(gate.worker, json.dumps(config))
    assert "error" in result and gate.worker.profiler is None
    assert not list(tmp_path.iterdir())


def test_rejects_existing_cpu_or_native_profiler(tmp_path):
    gate = setup(tmp_path)
    gate.worker._qfn_cpu_profile = object()
    assert "CPU profiler" in gate.start(gate.worker, gate.payload)["error"]
    gate.worker._qfn_cpu_profile = None
    assert "error" not in gate.start(gate.worker, gate.payload)
    assert "already exists" in gate.start(gate.worker, gate.payload)["error"]
    assert "already exists" in gate.start(gate.worker)["error"]
    assert "without an output" in gate.stop(gate.worker, "cpu-path")["error"]
    assert gate.stop(gate.worker)["ownership_released"]


def test_cpu_compatibility_and_unowned_stop(tmp_path):
    gate = setup(tmp_path)
    assert gate.start(gate.worker) == {"cpu": "start"}
    assert gate.stop(gate.worker, "cpu-path") == {"cpu": "stop"}
    gate.cpu_stop.assert_called_once_with(gate.worker, "cpu-path")
    assert "not started" in gate.stop(gate.worker)["error"]


def test_does_not_release_replaced_profiler(tmp_path):
    gate = setup(tmp_path)
    gate.start(gate.worker, gate.payload)
    replacement = object()
    gate.worker.profiler = replacement
    assert "ownership changed" in gate.stop(gate.worker)["error"]
    assert gate.worker.profiler is replacement


def test_factory_targets_only_two_cold_rpcs(monkeypatch):
    module = ModuleType("vllm_ascend._310p.worker_310p")
    module.NPUWorker310 = SimpleNamespace(qfn_cpu_profile_start=Mock(), qfn_cpu_profile_stop=Mock())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert set(candidate.replacements()) == {
        f"{candidate.WORKER_TARGET}.qfn_cpu_profile_start",
        f"{candidate.WORKER_TARGET}.qfn_cpu_profile_stop",
    }


def test_dedicated_rpcs_ignore_legacy_inventory_collision(tmp_path, monkeypatch):
    from tools.sidecard.qfn_collective_profile import QfnProfileWorkerExtension

    monkeypatch.setattr(candidate, "dependencies", lambda: (SimpleNamespace, FakeWrapper))
    worker = QfnProfileWorkerExtension()
    worker.rank = 0
    worker.profiler = None
    worker.qfn_cpu_profile_start = Mock(side_effect=AssertionError("legacy inventory RPC must not run"))
    worker.qfn_cpu_profile_stop = Mock(side_effect=AssertionError("legacy gate RPC must not run"))
    graphs, weights = object(), object()
    worker.graphs, worker.weights = graphs, weights
    payload = json.dumps(dict(output_dir=str(tmp_path), tag="dedicated", delay=0, max=2))
    assert worker.qfn_native_profile_status() == dict(rank=0, owned=False, profiler_present=False)
    assert worker.qfn_native_profile_start(payload)["running"]
    assert worker.qfn_native_profile_status()["owned"]
    assert worker.qfn_native_profile_stop()["ownership_released"]
    assert not worker.qfn_native_profile_status()["owned"]
    assert worker.graphs is graphs and worker.weights is weights
    worker.qfn_cpu_profile_start.assert_not_called()
    worker.qfn_cpu_profile_stop.assert_not_called()


def test_dedicated_rpcs_return_errors_instead_of_poisoning_executor_queues():
    from tools.sidecard.qfn_collective_profile import QfnProfileWorkerExtension

    worker = QfnProfileWorkerExtension()
    worker.rank = 3
    assert "error" in worker.qfn_native_profile_start()
    assert "error" in worker.qfn_native_profile_start("invalid json")
    assert "error" in worker.qfn_native_profile_stop()
