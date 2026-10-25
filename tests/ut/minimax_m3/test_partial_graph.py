# SPDX-License-Identifier: Apache-2.0
"""Ownership/admission regression tests; replay arithmetic is checked on 310P."""

import contextlib
import sys
import types
from unittest.mock import Mock

import pytest
import torch

from tools.minimax_m3.partial_graph import add_replacements
from tools.minimax_m3.resident import candidate_source


@pytest.fixture
def graph_runtime(monkeypatch):
    captures, graphs = [], []
    stream = types.SimpleNamespace(wait_stream=Mock(), synchronize=Mock())

    class Graph:
        def __init__(self):
            self.replay, self.reset = Mock(), Mock()
            graphs.append(self)

    @contextlib.contextmanager
    def capture(graph, *, pool, stream):
        captures.append((pool, stream))
        yield

    npu = types.SimpleNamespace(
        synchronize=Mock(),
        empty_cache=Mock(),
        graph_pool_handle=Mock(side_effect=object),
        current_stream=lambda: stream,
        Stream=lambda **kwargs: stream,
        stream=lambda value: contextlib.nullcontext(),
        NPUGraph=Graph,
        graph=capture,
        mem_get_info=lambda device: (2**31, 2**32),
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    vote = Mock(side_effect=lambda tensor, **kwargs: tensor.mul_(2))
    monkeypatch.setattr(torch.distributed, "all_reduce", vote)
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: types.SimpleNamespace(cpu_group="cpu", world_size=2)
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)

    class Attention:
        def _sparse_prepare(self, positions, hidden):
            return tuple(hidden.clone() for _ in range(5))

        def _run_sparse_attention(self, q, k, v, iq, ik, output):
            output.copy_(q)

        def o_proj(self, hidden):
            return hidden * 2, None

    class Layer:
        def __init__(self, index):
            self.layer_idx, self.self_attn, self.is_layer_sparse = index, Attention(), False
            self.input_layernorm = lambda h, r=None: h + 1 if r is None else (h + r, h + r)
            self.post_attention_layernorm = lambda h, r: (h + r, h + r)
            self.mlp = lambda h: h * 3
            self.fallbacks = 0

        def forward(self, positions, hidden, residual):
            self.fallbacks += 1
            return hidden, hidden if residual is None else residual

    class Model:
        def __init__(self):
            self.layers = [Layer(0), Layer(1)]

        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
            hidden, residual = inputs_embeds, None
            for layer in self.layers:
                hidden, residual = layer.forward(positions, hidden, residual)
            return hidden, residual

    class Worker:
        _m3_idle = Mock()

        def resident_apply(self, generation):
            return generation

        def resident_status(self):
            return {"rank": 0}

    model_module = types.ModuleType("vllm_ascend.models.minimax_m3.minimax_m3")
    model_module.MiniMaxM3DecoderLayer, model_module.MiniMaxM3Model = Layer, Model
    model_module.MiniMaxM3SparseAttention = Attention
    worker_module = types.ModuleType("vllm_ascend._310p.worker_310p")
    worker_module.NPUWorker310 = Worker
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    changes = add_replacements({})
    for name, value in changes.items():
        module, attribute = name.split(":")
        cls, method = attribute.split(".")
        monkeypatch.setattr(getattr(sys.modules[module], cls), method, value, raising=False)
    return types.SimpleNamespace(model=Model(), worker=Worker(), captures=captures, graphs=graphs, vote=vote, npu=npu)


def test_one_pool_direct_buffers_no_recapture_and_drained_cleanup(graph_runtime):
    runtime = graph_runtime
    with torch.inference_mode():
        first = runtime.model.forward(None, torch.tensor([7]), None, torch.zeros(1, 4))
        second = runtime.model.forward(None, torch.tensor([260]), None, torch.ones(1, 4))
    stats = runtime.worker.resident_capture()
    assert stats["piecewise_graph"]["captures"] == 4
    assert stats["piecewise_graph"]["replays"] == 8
    assert stats["piecewise_graph"]["copies"] == 4
    assert stats["direct_layers"] == 1
    assert first[0].data_ptr() == second[0].data_ptr()
    assert len({id(pool) for pool, stream in runtime.captures}) == 1
    assert len({id(stream) for pool, stream in runtime.captures}) == 1
    assert runtime.vote.call_count == 2  # Cold admission only, not every token.
    for call in runtime.vote.call_args_list:
        assert call.args[0].dtype == torch.float32
        assert call.kwargs["op"] == torch.distributed.ReduceOp.SUM
        assert call.kwargs["group"] == "cpu"
    assert runtime.worker.resident_apply("next") == "next"
    assert all(graph.reset.call_count == 1 for graph in runtime.graphs)


def test_peer_memory_pressure_prevents_capture_on_both_ranks(graph_runtime):
    runtime = graph_runtime
    runtime.vote.side_effect = lambda tensor, **kwargs: tensor.fill_(1)
    with torch.inference_mode():
        for _ in range(2):
            runtime.model.forward(None, torch.tensor([7]), None, torch.zeros(1, 4))
    assert not runtime.graphs
    assert runtime.vote.call_count == 2
    assert runtime.worker.resident_capture()["disabled_layers"] == 2
    assert all(layer.fallbacks == 2 for layer in runtime.model.layers)


def test_prefill_remains_eager(graph_runtime):
    with torch.inference_mode():
        graph_runtime.model.forward(None, torch.arange(4), None, torch.zeros(4, 4))
    assert not graph_runtime.captures
    graph_runtime.vote.assert_not_called()


def test_graph_source_is_snapshotted_and_keeps_native_replacements():
    source = 'def add_replacements(changes):\n    changes["probe"] = lambda: "captured"\n    return changes\n'
    module = types.ModuleType("composed")
    exec(candidate_source(native_name="int8", graph_source=source), module.__dict__)
    changes = module.replacements({"int8": lambda *args: "projection"})
    assert changes["probe"]() == "captured"
    assert changes["vllm_ascend._310p.ops.minimax_m3:grouped_w4_mm"](1, 2, 3, 4) == "projection"
