# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
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
    graphs.dsa_graphs = {}
    assert graphs.forward(state) == completed
    assert calls == ["replay", "remote", "dsa"]
    assert all(torch.equal(target, source) for target, source in zip(inputs, state, strict=True))


def test_prefill_uses_eager_routes_and_cache_updates():
    state = tuple(torch.ones(4, 2) for _ in range(4))
    calls = []
    graphs = object.__new__(ColdDecodeGraphs)
    graphs.layers = (SimpleNamespace(index=18, forward=lambda *args: calls.append(args) or args),)
    graphs.graphs = {18: (None, None, None)}
    graphs.dsa_graphs = {}
    assert graphs.forward(state) == state
    assert calls == [state]


def test_dsa_graph_routes_finish_outside_capture_and_prefill_stays_eager():
    state = tuple(torch.ones(1, 2) for _ in range(4))
    routes = (*state, state[0], state[0])
    calls = []
    graphs = object.__new__(ColdDecodeGraphs)
    graphs.layers = (
        SimpleNamespace(
            index=19,
            finish_forward=lambda *args: calls.append(("remote", args)) or args[:4],
            forward=lambda *args: calls.append(("eager", args)) or args,
        ),
    )
    graphs.graphs = {}
    graphs.dsa_graphs = {19: SimpleNamespace(forward=lambda value: calls.append(("graph", value)) or routes)}
    assert graphs.forward(state) == state
    assert [name for name, _ in calls] == ["graph", "remote"]
    assert calls[1][1] == routes
    large = tuple(x.expand(4, -1) for x in state)
    assert graphs.forward(large) == large
    assert calls[-1] == ("eager", large)
    graphs.close()
    assert graphs.dsa_graphs == {}


def diagnostic_fixture():
    calls = []

    class Observer:
        active = None

        def capture_begin(self, key):
            assert self.active is None
            self.active = key
            calls.append(("capture_begin", key))

        def capture_end(self, key):
            assert self.active == key
            self.active = None
            calls.append(("capture_end", key))

        def replayed(self, key):
            assert self.active is None
            calls.append(("observed", key))

    observer = Observer()

    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=allocate(torch.zeros),
        ones=allocate(torch.ones),
        eye=allocate(torch.eye),
        float16=torch.float16,
        float32=torch.float32,
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            CUDAGraph=lambda: SimpleNamespace(replay=lambda: calls.append(("replay", None))),
            graph=lambda _: nullcontext(),
        ),
    )

    def layer(index):
        return SimpleNamespace(
            index=index,
            linear=True,
            execution_diagnostic=observer,
            prepare_forward=lambda *state: calls.append(("prepare", observer.active)) or state,
            route=lambda x: (x, x),
            finish_forward=lambda *args: calls.append(("finish", index)) or args[:4],
            forward=lambda *state: calls.append(("eager", index)) or state,
            reset=lambda: None,
        )

    return fake, observer, calls, layer


def test_diagnostics_register_only_actual_capture_and_observe_before_experts():
    fake, observer, calls, layer = diagnostic_fixture()
    graphs = ColdDecodeGraphs([layer(18), layer(20)], fake)
    assert calls == [
        ("prepare", None),
        ("prepare", None),
        ("capture_begin", "cold-layer:18"),
        ("prepare", "cold-layer:18"),
        ("capture_end", "cold-layer:18"),
        ("prepare", None),
        ("prepare", None),
        ("capture_begin", "cold-layer:20"),
        ("prepare", "cold-layer:20"),
        ("capture_end", "cold-layer:20"),
    ]
    assert observer.active is None
    calls.clear()
    state = (
        torch.ones((1, 4096)).half(),
        torch.ones((1, 4, 4096)),
        torch.ones((1, 4, 1)),
        torch.eye(4)[None].contiguous(),
    )
    graphs.forward(state)
    assert calls == [
        ("replay", None),
        ("observed", "cold-layer:18"),
        ("finish", 18),
        ("replay", None),
        ("observed", "cold-layer:20"),
        ("finish", 20),
    ]
    calls.clear()
    graphs.forward(tuple(value.expand(4, *value.shape[1:]) for value in state))
    assert calls == [("eager", 18), ("eager", 20)]


def test_diagnostics_reject_multiple_owners_before_allocating_graphs():
    fake, _, calls, layer = diagnostic_fixture()
    first, second = layer(18), layer(20)
    second.execution_diagnostic = object()
    with pytest.raises(ValueError, match="diagnostic owners"):
        ColdDecodeGraphs([first, second], fake)
    assert not calls


def test_capture_registration_ends_on_capture_failure():
    fake, observer, calls, layer = diagnostic_fixture()
    first = layer(18)
    warmup = first.prepare_forward

    def fail_inside_capture(*state):
        if observer.active is not None:
            raise RuntimeError("capture failure")
        return warmup(*state)

    first.prepare_forward = fail_inside_capture
    with pytest.raises(RuntimeError, match="capture failure"):
        ColdDecodeGraphs([first], fake)
    assert observer.active is None
    assert calls[-2:] == [("capture_begin", "cold-layer:18"), ("capture_end", "cold-layer:18")]


def test_dsa_adapter_handles_its_own_observation_without_duplicate_cold_hook(monkeypatch):
    from tools.sidecard import cold_decode_graph

    fake, observer, calls, layer = diagnostic_fixture()
    dsa = layer(19)
    dsa.linear = False
    state = tuple(torch.ones((1, 2)) for _ in range(4))

    def make_dsa(actual, torch_stub, *, cold):
        assert actual is dsa and torch_stub is fake and cold

        def forward(values):
            observer.replayed("cuda-dsa:19:before")
            observer.replayed("cuda-dsa:19:after")
            return (*values, values[0], values[0])

        return SimpleNamespace(forward=forward)

    monkeypatch.setattr(cold_decode_graph, "DsaDecodeGraph", make_dsa)
    graphs = ColdDecodeGraphs([dsa], fake, dsa_graphs=True)
    assert not calls
    graphs.forward(state)
    assert calls == [
        ("observed", "cuda-dsa:19:before"),
        ("observed", "cuda-dsa:19:after"),
        ("finish", 19),
    ]
