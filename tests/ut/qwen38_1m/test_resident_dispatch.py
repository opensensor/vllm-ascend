# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The opt-in Qwen extension preserves graph calls and direct arguments."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load_extension(monkeypatch, base):
    class Wrapper:
        def __init__(self):
            self.runnable = Mock(return_value="direct")
            self.graph = Mock(return_value="graph")

        def __call__(self, *args, **kwargs):
            return self.graph(*args, **kwargs)

    worker_module = types.ModuleType("tools.glm_perf.resident_worker")
    worker_module.ResidentWorkerExtension = base
    graph_module = types.ModuleType("vllm_ascend.compilation.breakable_aclgraph")
    graph_module.BreakableACLGraphWrapper = Wrapper
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, graph_module.__name__, graph_module)
    path = Path(__file__).resolve().parents[3] / "tools/qwen4exp/resident_worker.py"
    spec = importlib.util.spec_from_file_location("qwen_resident_dispatch_test", path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    assert issubclass(extension.QwenResidentExtension, worker_module.ResidentWorkerExtension)
    return extension, Wrapper


def test_qwen_extension_dispatch(monkeypatch):
    _, Wrapper = load_extension(monkeypatch, type("ResidentWorkerExtension", (), {}))
    wrapper = Wrapper()
    assert wrapper(3, metadata="captured") == "graph"
    wrapper.graph.assert_called_once_with(3, metadata="captured")
    wrapper.runnable.assert_not_called()
    wrapper._resident_direct = True
    assert wrapper(6, metadata="live") == "direct"
    wrapper.runnable.assert_called_once_with(6, metadata="live")
    wrapper._resident_direct = False
    assert wrapper(3, metadata="captured") == "graph"
    assert wrapper.graph.call_count == 2


@pytest.mark.parametrize("has_tiers", [False, True])
def test_reset_clears_mamba_metadata_after_base_reset(monkeypatch, has_tiers):
    events = []

    class Base:
        def resident_reset(self):
            events.append("base_reset")
            return self.resident_status()

        def resident_status(self):
            return {"pid": 101, "graphs_dirty": False}

    extension, _ = load_extension(monkeypatch, Base)
    worker = extension.QwenResidentExtension()
    tiers = {1: Mock(), 2: Mock()} if has_tiers else {}
    for group_id, tier in tiers.items():
        tier.reset.side_effect = lambda group_id=group_id: events.append(group_id)
        tier.cache_status.return_value = {"host_checkpoints": 0}
    worker.model_runner = SimpleNamespace(_prefix_mamba_tiers=tiers) if has_tiers else SimpleNamespace()
    result = worker.resident_reset()
    assert events == ["base_reset", *tiers]
    assert result == {
        "pid": 101,
        "graphs_dirty": False,
        "prefix_mamba": {str(group_id): {"host_checkpoints": 0} for group_id in tiers},
    }


def test_pending_execution_prevents_mamba_reset(monkeypatch):
    class Base:
        def resident_reset(self):
            raise RuntimeError("pending model execution")

    extension, _ = load_extension(monkeypatch, Base)
    worker = extension.QwenResidentExtension()
    tier = Mock()
    worker.model_runner = SimpleNamespace(_prefix_mamba_tiers={1: tier})
    with pytest.raises(RuntimeError, match="pending model execution"):
        worker.resident_reset()
    tier.reset.assert_not_called()
