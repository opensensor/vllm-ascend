# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.expert_decode_graph import ExpertDecodeGraphs, prefill_graph_rows


def test_expert_capture_owns_banks_and_replays_only_single_rows():
    calls = []

    class Actor:
        rank = 1
        banks = {18: None, 19: None}

        def __call__(self, layer, x, ids, weights):
            calls.append((layer, x.shape[0]))
            return x.float()

    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=allocate(torch.zeros),
        full=allocate(torch.full),
        float16=torch.float16,
        float32=torch.float32,
        int32=torch.int32,
        npu=SimpleNamespace(
            synchronize=lambda: None,
            NPUGraph=lambda: SimpleNamespace(replay=lambda: calls.append("replay")),
            graph=lambda graph: nullcontext(),
        ),
    )
    native = Actor()
    actor = ExpertDecodeGraphs(native, fake)
    assert calls == [(18, 1)] * 3 + [(19, 1)] * 3
    x, ids, weights = torch.ones(1, 4096).half(), torch.full((1, 8), 155), torch.ones(1, 8)
    output = actor(18, x, ids, weights)
    assert output is actor.graphs[18][2] and calls[-1] == "replay"
    assert all(torch.equal(a, b) for a, b in zip(actor.graphs[18][1], (x, ids, weights), strict=True))
    output = actor(19, x.expand(4, -1), ids.expand(4, -1), weights.expand(4, -1))
    assert output.shape[0] == 4 and calls[-1] == (19, 4)
    actor.close()
    assert actor.graphs == {} and actor.actor is native


def test_prefill_capture_keeps_exact_shape_and_tail_fallback():
    calls = []
    actor = SimpleNamespace(rank=0, banks={18: None})

    class Native:
        rank, banks = actor.rank, actor.banks

        def __call__(self, layer, x, ids, weights):
            calls.append((layer, x.shape[0]))
            return x.float()

    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=allocate(torch.zeros),
        full=allocate(torch.full),
        float16=torch.float16,
        float32=torch.float32,
        int32=torch.int32,
        npu=SimpleNamespace(
            synchronize=lambda: None,
            NPUGraph=lambda: SimpleNamespace(replay=lambda: calls.append("replay")),
            graph=lambda graph: nullcontext(),
        ),
    )
    graphs = ExpertDecodeGraphs(Native(), fake, prefill_rows=32)
    assert calls == [(18, 1)] * 3 + [(18, 32)] * 3
    inputs = torch.ones(32, 4096).half(), torch.full((32, 8), 19), torch.ones(32, 8)
    assert graphs(18, *inputs) is graphs.prefill_graphs[18][2]
    assert calls[-1] == "replay"
    assert all(torch.equal(a, b) for a, b in zip(inputs, graphs.prefill_graphs[18][1], strict=True))
    graphs(18, *(v[:7] for v in inputs))
    assert calls[-1] == (18, 7)
    graphs.close()
    assert not graphs.graphs and not graphs.prefill_graphs


def test_prefill_graph_configuration_rejects_unsupported_geometry_before_capture():
    assert prefill_graph_rows({}) is None
    assert prefill_graph_rows({"direct_dma": {"npu_prefill_graphs": True, "npu_expert_graphs": True}}) == 32
    for spec in (
        {"direct_dma": {"npu_prefill_graphs": True}},
        {"direct_dma": {"npu_prefill_graphs": 1}},
        {
            "expert_transport": "direct-dma",
            "prefill_chunk_size": 64,
            "direct_dma": {"npu_prefill_graphs": True, "npu_expert_graphs": True},
        },
    ):
        with pytest.raises(ValueError):
            prefill_graph_rows(spec)
    for rows in (1, 17, 64, True):
        with pytest.raises(ValueError, match="exactly 32"):
            ExpertDecodeGraphs(None, None, prefill_rows=rows)
