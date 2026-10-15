# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import gdn_projection_gate as gate
from tools.qwen4exp.resident_candidates import qfn_gdn_projection_probe as probe


def layer():
    return SimpleNamespace(
        in_proj_qkv=torch.arange(32, dtype=torch.float16).reshape(4, 8) / 32,
        in_proj_z=torch.arange(16, dtype=torch.float16).reshape(2, 8) / 16,
        compute_dtype=torch.float32,
    )


def linear(inputs, weight, dtype):
    return torch.nn.functional.linear(inputs.to(weight.dtype), weight)


def prepared():
    owner = {}
    gate.prepare(layer(), linear, owner, format_weight=lambda value: value.clone())
    return owner


class FakeGraph:
    def __init__(self, npu):
        self.npu, self.operations, self.reset_count = npu, [], 0

    def replay(self):
        if self.npu.fail_replay:
            raise RuntimeError("replay submission failed")
        if self.npu.stale_replay:
            return
        for function, owner, inputs, outputs in self.operations:
            for output, actual in zip(outputs, function(owner, inputs), strict=True):
                output.copy_(actual)

    def reset(self):
        if self.npu.fail_reset:
            raise RuntimeError("reset failed")
        self.reset_count += 1


class FakeStream:
    def wait_stream(self, other):
        self.waited = other

    def synchronize(self):
        pass


class FakeNPU:
    def __init__(self):
        self.active = None
        self.current = FakeStream()
        self.fail_replay, self.stale_replay, self.fail_reset = False, False, False
        self.graphs = []

    def NPUGraph(self):
        graph = FakeGraph(self)
        self.graphs.append(graph)
        return graph

    def graph_pool_handle(self):
        return object()

    @contextlib.contextmanager
    def graph(self, graph, *, pool, stream):
        assert self.active is None and stream is self.current
        self.active = graph
        try:
            yield
        finally:
            self.active = None

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def current_stream(self, device=None):
        return self.current

    def Stream(self, device=None):
        return FakeStream()

    def synchronize(self, device=None):
        pass

    def mem_get_info(self):
        return 2 << 30, 48 << 30

    def is_current_stream_capturing(self):
        return self.active is not None


def fake_backend(monkeypatch):
    npu = FakeNPU()
    for name in ("baseline", "concatenated"):
        function = getattr(gate, name)

        def recorded(owner, inputs, function=function):
            result = function(owner, inputs)
            if npu.active is not None:
                npu.active.operations.append((function, owner, inputs, result))
            return result

        monkeypatch.setattr(gate, name, recorded)
    return SimpleNamespace(npu=npu, empty=torch.empty, inference_mode=torch.inference_mode)


def test_exact_gate_changing_inputs_graph_replay_and_owned_cleanup(monkeypatch):
    backend, owner = fake_backend(monkeypatch), prepared()
    caller = backend.npu.current
    result = gate.validate(owner, repeats=1, replays=1, torch_module=backend)
    assert result["passed"] and result["weights_preserved"]
    assert result["reset_graphs"] == 12
    assert len(result["checks"]) == 48 and len(result["timings"]) == 12
    assert {(item["rows"], item["mode"], item["step"], item["input_dtype"]) for item in result["checks"]} == {
        (row, mode, step, str(dtype))
        for row in gate.ROWS
        for mode in ("eager", "graph")
        for step in range(4)
        for dtype in gate.INPUT_DTYPES
    }
    assert all(graph.reset_count == 1 for graph in owner["graphs"])
    assert len({id(pool) for pool in owner["pools"]}) == 12
    assert backend.npu.current is caller and owner["stream"].waited is caller
    assert owner["validated"] and not owner["failed"]
    with pytest.raises(RuntimeError, match="one-shot"):
        gate.validate(owner, torch_module=backend)


@pytest.mark.parametrize("failure", ["stale_replay", "fail_replay", "fail_reset"])
def test_graph_failure_keeps_graphs_pools_weights_and_restores_caller(monkeypatch, failure):
    backend, owner = fake_backend(monkeypatch), prepared()
    caller = backend.npu.current
    setattr(backend.npu, failure, True)
    with pytest.raises(RuntimeError):
        gate.validate(owner, rows=(3,), repeats=1, replays=1, torch_module=backend)
    assert owner["failed"] and not owner.get("validated")
    expected_count = 4 if failure == "fail_reset" else 2
    assert len(owner["graphs"]) == expected_count and len(owner["pools"]) == expected_count
    assert all(graph.reset_count == 0 for graph in owner["graphs"])
    assert owner["packed"] is not None and owner["tensors"] and owner["outputs"]
    assert backend.npu.current is caller


def test_eager_mismatch_prevents_any_capture(monkeypatch):
    backend, owner = fake_backend(monkeypatch), prepared()
    with torch.inference_mode():
        owner["packed"][0, 2] += 1
    with pytest.raises(RuntimeError, match="no capture attempted"):
        gate.validate(owner, rows=(3,), repeats=1, replays=1, torch_module=backend)
    assert owner["failed"] and owner["graphs"] == []


def test_bitwise_comparison_distinguishes_signed_zero():
    assert not gate.outputs_exact(
        (torch.tensor([0.0], dtype=torch.float16),), (torch.tensor([-0.0], dtype=torch.float16),)
    )


def test_split_materialization_preserves_values_and_original_weights():
    owner = prepared()
    source = gate.input_case(6, 2, width=8)
    actual = gate.concatenated(owner, source)
    assert gate.outputs_exact(actual, gate.baseline(owner, source))
    assert tuple(value.stride() for value in actual) == ((4, 1), (2, 1))
    assert gate.weights_preserved(owner)
    owner["qkv"][0, 0] += 1
    assert not gate.weights_preserved(owner)


def request(action, **kwargs):
    return probe.parse_request(json.dumps(dict(action=action, **kwargs)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": "bind"},
        {"action": "preflight", "build": "/x"},
        {"action": "validate", "rows": []},
        {"action": "validate", "rows": [3, 3]},
        {"action": "validate", "rows": [True]},
        {"action": "validate", "rows": [2048]},
        {"action": "validate", "module": ""},
        {"action": "validate", "module": 4},
        {"action": "validate", "repeats": True},
        {"action": "validate", "repeats": 6},
        {"action": "validate", "replays": 0},
        {"action": "validate", "replays": 11},
    ],
)
def test_rpc_request_is_bounded(kwargs):
    with pytest.raises(ValueError):
        probe.parse_request(json.dumps(kwargs))


def worker_fixture(monkeypatch):
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(execute_model_state=None))
    resident = layer()
    monkeypatch.setattr(gate, "resident_module", lambda *args: ("layers.0.attention", resident, linear))
    monkeypatch.setattr(probe, "_format_weight", lambda value: value.clone())
    backend = fake_backend(monkeypatch)
    return worker, backend


def test_rpc_lifecycle_requires_separate_preflight_and_preserves_identity(monkeypatch):
    worker, backend = worker_fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="preflight"):
        probe.run_action(worker, request("load"), backend)
    assert probe.run_action(worker, request("preflight"), backend)["passed"]
    assert probe.run_action(worker, request("load"), backend)["passed"]
    state = getattr(worker, probe.STATE_ATTRIBUTE)
    owner = state["resources"]
    assert probe.run_action(worker, request("load"), backend)["already_loaded"]
    assert state["resources"] is owner
    with pytest.raises(RuntimeError, match="do not replace"):
        probe.run_action(worker, request("load", module="other"), backend)
    result = probe.run_action(worker, request("validate", repeats=1, replays=1), backend)
    assert result["passed"] and state["receipt"] is not None
    assert not state["failed"]


def test_allocation_denied_when_current_memory_falls_below_preflight(monkeypatch):
    worker, backend = worker_fixture(monkeypatch)
    probe.run_action(worker, request("preflight"), backend)
    backend.npu.mem_get_info = lambda: (gate.MEMORY_REQUIRED_BYTES - 1, 48 << 30)
    assert not probe.run_action(worker, request("load"), backend)["passed"]
    assert getattr(worker, probe.STATE_ATTRIBUTE)["resources"] == {}


def test_rpc_failure_retains_partial_tensor_traceback_and_no_retry(monkeypatch):
    worker, backend = worker_fixture(monkeypatch)
    probe.run_action(worker, request("preflight"), backend)
    references = []

    def fail(weight):
        temporary = torch.ones(10)
        references.append(weakref.ref(temporary))
        raise RuntimeError("format failed before return")

    monkeypatch.setattr(probe, "_format_weight", fail)
    with pytest.raises(RuntimeError, match="format failed"):
        probe.run_action(worker, request("load"), backend)
    gc.collect()
    state = getattr(worker, probe.STATE_ATTRIBUTE)
    assert references[0]() is not None and state["failure_traceback"] is not None
    assert state["resources"]["packed_nd"] is not None
    with pytest.raises(RuntimeError, match="retain all owners"):
        probe.run_action(worker, request("load"), backend)


@pytest.mark.parametrize("field", ["profiler", "_qfn_cpu_profile", "_qfn_native_profile_session"])
def test_profiling_blocks_probe(monkeypatch, field):
    worker, backend = worker_fixture(monkeypatch)
    setattr(worker, field, object())
    with pytest.raises(RuntimeError, match="profilers"):
        probe.run_action(worker, request("preflight"), backend)
    assert not hasattr(worker, probe.STATE_ATTRIBUTE)


def test_running_request_blocks_probe(monkeypatch):
    worker, backend = worker_fixture(monkeypatch)
    worker.model_runner.execute_model_state = object()
    with pytest.raises(RuntimeError, match="drain"):
        probe.run_action(worker, request("preflight"), backend)


def test_rpc_delegates_non_json_and_exposes_only_stop():
    rpc = probe.make_rpc(lambda worker, output=None: ("original", output))
    assert rpc(SimpleNamespace(), "output.json") == ("original", "output.json")
    assert rpc(SimpleNamespace(), "{invalid")["passed"] is False
    assert probe.RPC_TARGET.endswith(":NPUWorker310.qfn_cpu_profile_stop")


def test_resident_type_source_and_geometry_admission(monkeypatch):
    class ExactLayer:
        pass

    class DerivedLayer(ExactLayer):
        pass

    device = SimpleNamespace(type="npu")
    tensor = lambda shape: SimpleNamespace(shape=shape, dtype=torch.float16, device=device)
    value = ExactLayer()
    value.in_proj_qkv, value.in_proj_z = tensor((2560, 2560)), tensor((1536, 2560))
    value.params_dtype, value.compute_dtype, value.tp_size = torch.float16, torch.float32, 4
    model = SimpleNamespace(named_modules=lambda: (("derived", DerivedLayer()), ("exact", value)))
    worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))
    operand = object()
    module = SimpleNamespace(_GDNAttention=ExactLayer, _linear=linear, _linear_operand_dtype=operand)
    hashes = {ExactLayer: gate.CLASS_SHA256, linear: gate.LINEAR_SHA256, operand: gate.OPERAND_SHA256}
    monkeypatch.setattr(gate, "source_hash", hashes.__getitem__)
    assert gate.resident_module(worker, None, model_module=module) == ("exact", value, linear)
    with pytest.raises(ValueError, match="exact pinned"):
        gate.resident_module(worker, "derived", model_module=module)
    value.tp_size = 2
    with pytest.raises(RuntimeError, match="TP4"):
        gate.resident_module(worker, None, model_module=module)
    hashes[linear] = "changed"
    with pytest.raises(RuntimeError, match="source"):
        gate.resident_module(worker, None, model_module=module)


def test_resident_inference_weights_have_no_version_counter():
    with torch.inference_mode():
        resident = layer()
    owner = {}
    gate.prepare(resident, linear, owner, format_weight=lambda value: value.clone())
    assert owner["weight_versions"] == (None, None)
    assert gate.weights_preserved(owner)


def test_production_input_retains_non_half_exact_values():
    inputs = gate.input_case(3, 0, width=8, dtype=torch.float32)
    assert inputs.dtype == torch.float32
    assert not torch.equal(inputs, inputs.half().float())
    assert torch.signbit(inputs[:, 0]).all()
    owner = prepared()
    assert gate.outputs_exact(gate.concatenated(owner, inputs), gate.baseline(owner, inputs))
