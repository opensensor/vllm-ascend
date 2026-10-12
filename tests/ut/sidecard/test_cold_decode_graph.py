# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from tools.sidecard.cold_decode_graph import ColdDecodeGraphs


def test_capture_never_calls_remote_experts_and_resets_recurrent_state():
    calls = []

    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=allocate(torch.zeros),
        ones=allocate(torch.ones),
        eye=allocate(torch.eye),
        float16=torch.float16,
        float32=torch.float32,
        cuda=SimpleNamespace(synchronize=lambda: None, CUDAGraph=lambda: object(), graph=lambda _: nullcontext()),
    )
    layers = []
    for index, linear in ((18, True), (19, False)):
        layers.append(
            SimpleNamespace(
                index=index,
                linear=linear,
                prepare_forward=lambda *state: calls.append("prepare") or state,
                route=lambda x: (x, x),
                finish_forward=lambda *args: calls.append("remote"),
                reset=lambda: calls.append("reset"),
            )
        )
    graphs = ColdDecodeGraphs(layers, fake)
    assert list(graphs.graphs) == [18]
    assert calls == ["prepare"] * 3 + ["reset"] * 2


def test_decode_passes_captured_routes_to_experts_before_next_attention():
    calls = []
    inputs = tuple(torch.zeros(1, 2) for _ in range(4))
    outputs = tuple(torch.full((1, 2), float(index)) for index in range(6))
    state = tuple(torch.ones(1, 2) for _ in range(4))
    completed = tuple(torch.full((1, 2), 9.0) for _ in range(4))

    def finish(*args):
        assert all(a is b for a, b in zip(args, outputs, strict=True))
        calls.append("remote")
        return completed

    def eager(*args):
        assert all(a is b for a, b in zip(args, completed, strict=True))
        calls.append("dsa")
        return args

    graphs = object.__new__(ColdDecodeGraphs)
    graphs.layers = (
        SimpleNamespace(index=18, finish_forward=finish),
        SimpleNamespace(index=19, forward=eager),
    )
    graphs.graphs = {18: (SimpleNamespace(replay=lambda: calls.append("replay")), inputs, outputs)}
    assert graphs.forward(state) == completed
    assert calls == ["replay", "remote", "dsa"]
    assert all(torch.equal(target, source) for target, source in zip(inputs, state, strict=True))


def test_prefill_uses_eager_routes_and_cache_updates():
    state = tuple(torch.ones(4, 2) for _ in range(4))
    calls = []
    graphs = object.__new__(ColdDecodeGraphs)
    graphs.layers = (SimpleNamespace(index=18, forward=lambda *args: calls.append(args) or args),)
    graphs.graphs = {18: (None, None, None)}
    assert graphs.forward(state) == state
    assert calls == [state]
