# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.layer_decode_graph import DecodeGraphs, forward_span, linear_spans


def test_middle_spans_keep_dsa_eager():
    layers = [SimpleNamespace(index=i, linear=i not in (19, 23, 27, 31)) for i in range(18, 33)]
    spans = linear_spans(layers)
    assert [[layer.index for layer in span] for span in spans] == [
        [18],
        [19],
        [20, 21, 22],
        [23],
        [24, 25, 26],
        [27],
        [28, 29, 30],
        [31],
        [32],
    ]
    assert linear_spans([]) == ()


def test_forward_span_keeps_all_deferred_fields():
    seen = []

    def forward(*state):
        seen.append(state)
        return tuple(value + 1 for value in state)

    layer = SimpleNamespace(forward=forward)
    assert forward_span((layer, layer), (1, 2, 3, 4)) == (3, 4, 5, 6)
    assert seen == [(1, 2, 3, 4), (2, 3, 4, 5)]


def test_prefill_bypasses_capture_and_decode_copies_every_input():
    class Value:
        def __init__(self, rows):
            self.shape = (rows,)
            self.copied = None

        def copy_(self, source):
            self.copied = source

    calls = []
    layer = SimpleNamespace(index=18, forward=lambda *state: calls.append(state) or state)
    executor = object.__new__(DecodeGraphs)
    executor.spans = ((layer,),)
    inputs = tuple(Value(1) for _ in range(4))
    outputs = tuple(Value(1) for _ in range(4))
    graph = SimpleNamespace(replay=lambda: calls.append("replay"))
    executor.graphs = {18: (graph, inputs, outputs)}
    prefill = tuple(Value(32) for _ in range(4))
    assert executor.forward(prefill) == prefill
    assert calls == [prefill]
    decode = tuple(Value(1) for _ in range(4))
    assert executor.forward(decode) is outputs
    assert calls == [prefill, "replay"]
    assert all(target.copied is source for target, source in zip(inputs, decode, strict=True))


def test_capture_warms_linear_spans_and_resets_persistent_state():
    calls = []

    def layer(index, linear):
        return SimpleNamespace(
            index=index,
            linear=linear,
            forward=lambda *state: calls.append((index, "forward")) or state,
            reset=lambda: calls.append((index, "reset")),
        )

    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=allocate(torch.zeros),
        ones=allocate(torch.ones),
        eye=allocate(torch.eye),
        float16=torch.float16,
        float32=torch.float32,
        npu=SimpleNamespace(synchronize=lambda: None, NPUGraph=lambda: object(), graph=lambda _: nullcontext()),
    )
    layers = [layer(18, True), layer(19, False), layer(20, True)]
    executor = DecodeGraphs(layers, fake, "npu:0")
    assert list(executor.graphs) == [18, 20]
    assert calls.count((18, "forward")) == 3
    assert calls.count((20, "forward")) == 3
    assert (19, "forward") not in calls
    assert calls[-3:] == [(18, "reset"), (19, "reset"), (20, "reset")]


def test_initial_cuda_span_rejects_unexpected_deferred_state():
    executor = object.__new__(DecodeGraphs)
    executor.spans = ((SimpleNamespace(index=0),),)
    replay = SimpleNamespace(replay=lambda: pytest.fail("invalid state replayed"))
    executor.graphs = {0: (replay, (torch.zeros(1, 4096), None, None, None), None)}
    with pytest.raises(ValueError, match="empty deferred state"):
        executor.forward((torch.ones(1, 4096), object(), None, None))
