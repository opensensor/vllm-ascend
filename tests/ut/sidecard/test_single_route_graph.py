# SPDX-License-Identifier: Apache-2.0
"""One owned contribution selects the fast graph without any tensor readback."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.single_route_gate import validate
from tools.sidecard.single_route_graph import SingleRouteDecodeGraphs, single_route_enabled


def fixture():
    owner = SimpleNamespace(rank=0, banks={18: None}, stable_rows=17)

    class Actor:
        rank, banks = owner.rank, owner.banks
        stable_rows = owner.stable_rows

        def __call__(self, layer, x, ids, weights):
            return x.float() + self.stable_rows

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
            NPUGraph=lambda: SimpleNamespace(replay=lambda: None),
            graph=lambda graph: nullcontext(),
        ),
    )
    return Actor(), fake


def test_single_route_capture_restores_padding_and_selection_requires_exact_count():
    owner, fake = fixture()
    graphs = SingleRouteDecodeGraphs(owner, fake)
    assert owner.stable_rows == 17
    inputs = torch.ones(1, 4096).half(), torch.zeros(1, 8, dtype=torch.int32), torch.ones(1, 8)
    assert graphs(18, *inputs, owned_routes=1) is graphs.single_route_graphs[18][2]
    for count in (None, 0, 2, 8):
        assert graphs(18, *inputs, owned_routes=count) is graphs.graphs[18][2]
    for count in (-1, 9, True, 1.0, "1"):
        with pytest.raises(ValueError, match="owned route count"):
            graphs(18, *inputs, owned_routes=count)
    multi = tuple(value.expand(4, -1) for value in inputs)
    assert torch.equal(graphs(18, *multi, owned_routes=1), multi[0].float() + 17)
    graphs.close()
    assert not graphs.single_route_graphs and not graphs.graphs


def test_single_route_capture_restores_padding_when_capture_fails(monkeypatch):
    owner, fake = fixture()
    original = SingleRouteDecodeGraphs.capture

    def capture(actor, torch, layer, rows):
        if actor.stable_rows == 1:
            raise RuntimeError("failed capture")
        return original(actor, torch, layer, rows)

    monkeypatch.setattr(SingleRouteDecodeGraphs, "capture", staticmethod(capture))
    with pytest.raises(RuntimeError, match="failed capture"):
        SingleRouteDecodeGraphs(owner, fake)
    assert owner.stable_rows == 17


@pytest.mark.parametrize("value", [1, "true", None])
def test_single_route_requires_boolean_and_decode_capture(value):
    with pytest.raises(ValueError):
        single_route_enabled({"direct_dma": {"npu_single_route_graphs": value, "npu_expert_graphs": True}})


def test_single_route_configuration_defaults_off_and_requires_base_graphs():
    assert not single_route_enabled({})
    with pytest.raises(ValueError):
        single_route_enabled({"direct_dma": {"npu_single_route_graphs": True}})
    assert single_route_enabled({"direct_dma": {"npu_single_route_graphs": True, "npu_expert_graphs": True}})


@pytest.mark.parametrize("failure", [None, "rounding", "nonfinite"])
def test_gate_checks_every_layer_slot_and_scale_and_records_failure(tmp_path, failure):
    class Graphs:
        actor = SimpleNamespace(rank=1, banks={18: None, 19: None})

        def __call__(self, layer, hidden, ids, weights, *, owned_routes=None):
            assert int(((ids >= 144) & (weights > 0)).sum()) == 1
            value = hidden.float()
            if owned_routes == 1 and failure:
                value = value + 0.001 if failure == "rounding" else value * float("nan")
            return value

    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        manual_seed=torch.manual_seed,
        int32=torch.int32,
        randn=allocate(torch.randn),
        arange=allocate(torch.arange),
        full=allocate(torch.full),
    )
    output = tmp_path / "gate.json"
    if failure:
        with pytest.raises(AssertionError, match="expert output"):
            validate(Graphs(), fake, output)
    else:
        validate(Graphs(), fake, output)
    report = json.loads(output.read_bytes())
    assert report["passed"] == (failure is None)
    assert len(report["cases"]) == (64 if failure is None else 1)
