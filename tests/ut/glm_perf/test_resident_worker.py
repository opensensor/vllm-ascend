"""Exercise the shipped worker and audit extension using CPU graph stand-ins."""

import dataclasses
import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tools.glm_perf.resident_control import Control

ROOT = Path(__file__).resolve().parents[3]


@dataclasses.dataclass
class GraphParameters:
    events: dict
    workspaces: dict
    handles: dict
    attn_params: dict


def load_file(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(monkeypatch):
    class Wrapper:
        def __init__(self, runnable):
            self.runnable = runnable
            self.entries = {}
            self.graph_pool = "initial_pool"

        def clear_graphs(self):
            self.entries.clear()

        def __call__(self):
            if not self.entries:
                entry = types.SimpleNamespace(batch_descriptor="test", input_addresses=[])
                self.entries["test"] = entry
                return self._capture(entry, (), {})
            return self._replay(self.entries["test"], (), {})

        def _capture(self, entry, args, kwargs):
            entry.output = self.runnable()
            return entry.output

        def _replay(self, entry, args, kwargs):
            return entry.output

        def _collect_tensor_addresses(self, args, kwargs):
            return []

    parameters = GraphParameters({2: [object()]}, {2: object()}, {2: [object()]}, {2: [object()]})
    graph_module = types.ModuleType("vllm_ascend.compilation.acl_graph")
    for name in ("get_graph_params", "get_draft_graph_params", "get_draft_graph_prefill_params"):
        setattr(graph_module, name, lambda: parameters)
    wrapper_module = types.ModuleType("vllm_ascend.compilation.breakable_aclgraph")
    wrapper_module.BreakableACLGraphWrapper = Wrapper
    monitor = types.ModuleType("vllm.compilation.monitor")
    monitor.set_cudagraph_capturing_enabled = Mock()
    context = types.ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={})
    for module in (graph_module, wrapper_module, monitor, context):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(synchronize=Mock(), empty_cache=Mock(), graph_pool_handle=Mock(side_effect=object)),
        raising=False,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    extension = load_file("tools.glm_perf.resident_worker", ROOT / "tools/glm_perf/resident_worker.py", monkeypatch)
    audit = load_file(
        "resident_audit_test",
        ROOT / "artifacts/glm-perf-310p/mtp-packed-candidate-20261004/audit_graph_metadata.py",
        monkeypatch,
    )
    instance = audit.MetadataAuditExtension()
    weights = torch.nn.Parameter(torch.tensor([10.0]))
    projection = types.ModuleType("vllm_ascend._resident_projection")
    projection.forward = lambda value: value
    monkeypatch.setitem(sys.modules, projection.__name__, projection)

    class Model(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.weight = weights
            self.offset = offset

        def forward(self):
            return projection.forward(self.weight) + self.offset

    target = Wrapper(Model(1))
    draft = Wrapper(Model(2))
    cache = torch.ones(4)
    batch = types.SimpleNamespace(req_ids=["finished"], remove_request=Mock())
    runner = types.SimpleNamespace(
        model=target,
        drafter=types.SimpleNamespace(model=draft),
        use_async_scheduling=False,
        execute_model_state=None,
        input_batch=batch,
        requests={"finished": object()},
        kv_caches=[(cache, cache)],
    )

    def capture():
        assert not target._resident_direct and not draft._resident_direct
        target()
        draft()

    runner.capture_model = Mock(side_effect=capture)
    instance.model_runner = runner
    return types.SimpleNamespace(
        instance=instance,
        runner=runner,
        weights=weights,
        cache=cache,
        parameters=parameters,
        extension=extension,
        monitor=monitor,
    )


def apply(worker, setting):
    worker.instance.resident_prepare(json.dumps(dataclasses.asdict(setting)))
    worker.instance.resident_apply(setting.generation)
    return worker.instance.resident_capture()


def test_mode_switch_preserves_captures_and_selects_target_and_draft_explicitly(worker):
    target, draft = worker.runner.model, worker.runner.drafter.model
    target()
    draft()
    captures = (target.entries["test"], draft.entries["test"])
    for mode, expected in (
        ("direct-target", (True, False)),
        ("direct-draft", (False, True)),
        ("direct-both", (True, True)),
        ("graph", (False, False)),
    ):
        apply(worker, Control(uuid.uuid4().hex, mode))
        assert (target._resident_direct, draft._resident_direct) == expected
        assert (target.entries["test"], draft.entries["test"]) == captures
    worker.runner.capture_model.assert_not_called()
    worker.extension.torch.npu.empty_cache.assert_not_called()


def test_recapture_clears_attention_handles_and_retains_weight_and_cache_allocations(worker):
    weight_pointer, cache_pointer = worker.weights.data_ptr(), worker.cache.data_ptr()
    setting = Control(uuid.uuid4().hex, "direct-draft", recapture=True)
    receipt = apply(worker, setting)
    assert not receipt["graphs_dirty"]
    worker.runner.capture_model.assert_called_once()
    assert worker.parameters.events == {2: []}
    assert worker.parameters.handles == {2: []}
    assert worker.parameters.workspaces == {2: None}
    assert worker.runner.drafter.model._resident_direct
    worker.instance.resident_reset()
    assert worker.weights.data_ptr() == weight_pointer
    assert worker.cache.data_ptr() == cache_pointer
    torch.testing.assert_close(worker.weights, torch.tensor([10.0]))
    assert not torch.count_nonzero(worker.cache)
    assert not worker.runner.requests
    worker.runner.input_batch.remove_request.assert_called_once_with("finished")


def test_recapture_returns_retired_cached_blocks_before_capture_without_relocating_live_tensors(worker):
    actions = []
    worker.extension.torch.npu.empty_cache.side_effect = lambda: actions.append("release_cached")
    capture = worker.runner.capture_model.side_effect

    def check_capture():
        assert actions == ["release_cached"]
        actions.append("capture")
        capture()

    worker.runner.capture_model.side_effect = check_capture
    weight_pointer, cache_pointer = worker.weights.data_ptr(), worker.cache.data_ptr()
    receipt = apply(worker, Control(uuid.uuid4().hex, "graph", recapture=True))
    assert not receipt["graphs_dirty"]
    assert actions == ["release_cached", "capture"]
    assert worker.weights.data_ptr() == weight_pointer
    assert worker.cache.data_ptr() == cache_pointer
    torch.testing.assert_close(worker.weights, torch.tensor([10.0]))
    torch.testing.assert_close(worker.cache, torch.ones(4))


def test_recapture_renews_retired_allocator_pool_and_shares_it_between_models(worker):
    pools = []
    for _ in range(2):
        apply(worker, Control(uuid.uuid4().hex, recapture=True))
        target, draft = worker.runner.model, worker.runner.drafter.model
        assert target.graph_pool is draft.graph_pool
        assert target.graph_pool != "initial_pool"
        pools.append(target.graph_pool)
    assert pools[0] is not pools[1]
    assert torch.npu.graph_pool_handle.call_count == 2


def test_failed_capture_disables_capture_and_retains_dirty_generation(worker):
    worker.runner.capture_model.side_effect = RuntimeError("capture failed")
    receipt = apply(worker, Control(uuid.uuid4().hex, "direct-both", recapture=True))
    assert receipt["error"] == "RuntimeError: capture failed"
    assert worker.instance.resident_status()["graphs_dirty"]
    worker.monitor.set_cudagraph_capturing_enabled.assert_called_with(False)
    assert worker.runner.model._resident_direct


def test_storage_fingerprint_covers_packed_experts_outside_registered_parameters(worker):
    model = worker.runner.model.runnable
    model.w2_experts = [types.SimpleNamespace(gate_packed=torch.ones(4, dtype=torch.uint8))]
    before = worker.instance.resident_status()["weight_storage_digest"]
    previous = model.w2_experts[0].gate_packed
    model.w2_experts[0].gate_packed = previous.clone()
    assert worker.instance.resident_status()["weight_storage_digest"] != before


def test_python_candidate_changes_recaptured_output_and_baseline_restores_it(worker):
    original = worker.instance.resident_status()["weight_storage_digest"]
    source = """def forward(value):
    return value * 2
def replacements():
    return {"vllm_ascend._resident_projection:forward": forward}
"""
    receipt = apply(worker, Control(uuid.uuid4().hex, "graph", "double", source))
    torch.testing.assert_close(worker.runner.model(), torch.tensor([21.0]))
    torch.testing.assert_close(worker.runner.drafter.model(), torch.tensor([22.0]))
    assert receipt["weight_storage_digest"] == original
    receipt = apply(worker, Control(uuid.uuid4().hex))
    torch.testing.assert_close(worker.runner.model(), torch.tensor([11.0]))
    assert receipt["weight_storage_digest"] == original
    assert worker.runner.capture_model.call_count == 2


def test_reset_rejects_pending_execution_before_erasing_state(worker):
    worker.runner.execute_model_state = object()
    with pytest.raises(RuntimeError, match="pending"):
        worker.instance.resident_reset()
    torch.testing.assert_close(worker.cache, torch.ones(4))


def test_unprepared_apply_returns_error_receipt_without_poisoning_rpc_queue(worker):
    result = worker.instance.resident_apply(uuid.uuid4().hex)
    assert result["rank"] == 0
    assert "not been prepared" in result["error"]
    assert worker.instance.resident_status()["mode"] == "graph"


def test_invalid_prepare_returns_error_receipt_without_mutating_session(worker):
    result = worker.instance.resident_prepare('{"generation":"invalid"}')
    assert result["rank"] == 0 and "error" in result
    assert worker.instance._resident_session().current is None
