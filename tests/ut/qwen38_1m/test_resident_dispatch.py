# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The opt-in Qwen extension preserves graph calls and direct arguments."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


def test_qwen_extension_dispatch(monkeypatch):
    class Wrapper:
        def __init__(self):
            self.runnable = Mock(return_value="direct")
            self.graph = Mock(return_value="graph")

        def __call__(self, *args, **kwargs):
            return self.graph(*args, **kwargs)

    worker_module = types.ModuleType("tools.glm_perf.resident_worker")
    worker_module.ResidentWorkerExtension = type("ResidentWorkerExtension", (), {})
    graph_module = types.ModuleType("vllm_ascend.compilation.breakable_aclgraph")
    graph_module.BreakableACLGraphWrapper = Wrapper
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, graph_module.__name__, graph_module)
    path = Path(__file__).resolve().parents[3] / "tools/qwen4exp/resident_worker.py"
    spec = importlib.util.spec_from_file_location("qwen_resident_dispatch_test", path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    assert issubclass(extension.QwenResidentExtension, worker_module.ResidentWorkerExtension)
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
