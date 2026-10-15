# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A failed pre-reset drain must not mutate request or device state."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.qwen4exp.resident_candidates import qfn_reset_barrier as candidate


def install(monkeypatch, *, failed_drain=False, failed_reset=False):
    events = []
    result = {"rank": 0, "graphs_dirty": False}

    def synchronize():
        events.append("drain")
        if failed_drain:
            raise RuntimeError("device drain failed")

    def original_reset(worker):
        events.append(("reset", worker))
        if failed_reset:
            raise RuntimeError("reset failed")
        return result

    def error_result(error):
        events.append(("error", str(error)))
        return {"rank": 0, "error": str(error)}

    def wrong_dispatch():
        pytest.fail("wrapper recursively dispatched through the worker")

    monkeypatch.setattr(candidate.torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    module = ModuleType("tools.sidecard.qfn_resident_worker")
    module.QfnResidentWorkerExtension = SimpleNamespace(resident_reset=original_reset)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    worker = SimpleNamespace(_resident_error=error_result, resident_reset=wrong_dispatch)
    return worker, events, result


def test_drain_precedes_original_reset_without_recursive_dispatch(monkeypatch):
    worker, events, result = install(monkeypatch)
    assert candidate.resident_reset(worker) is result
    assert events == ["drain", ("reset", worker)]


def test_failed_drain_returns_rank_error_without_reset(monkeypatch):
    worker, events, _ = install(monkeypatch, failed_drain=True)
    assert candidate.resident_reset(worker) == {"rank": 0, "error": "device drain failed"}
    assert events == ["drain", ("error", "device drain failed")]


def test_unexpected_original_reset_error_is_acknowledged(monkeypatch):
    worker, events, _ = install(monkeypatch, failed_reset=True)
    assert candidate.resident_reset(worker) == {"rank": 0, "error": "reset failed"}
    assert events == ["drain", ("reset", worker), ("error", "reset failed")]


def test_only_actual_qualified_worker_method_is_replaced():
    assert candidate.replacements() == {
        "vllm_ascend._310p.worker_310p:NPUWorker310.resident_reset": candidate.resident_reset
    }
