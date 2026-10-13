# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.dsa_decode_graph import DsaDecodeGraph


@pytest.mark.parametrize("cold", [False, True])
def test_capture_never_appends_cache_or_selects_sparse_keys_or_dispatches_cold(cold):
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
    layer = SimpleNamespace(
        linear=False,
        device="cuda",
        tp_rank=None,
        config={"num_attention_heads": 2, "kv_lora_rank": 8},
        prepare_attention=lambda *state: calls.append("mhc attention") or state,
        dsa_prepare=lambda x: calls.append("qkv") or (x, x, x),
        dsa_project=lambda x: calls.append("output projection") or x,
        finish_attention=lambda *state: calls.append("mhc ffn") or state,
        finish_forward=lambda *state: calls.append("local experts") or state,
        route=lambda x: calls.append("route") or (x, x),
        dsa_attend=lambda *args: pytest.fail("cache and attention ran during capture"),
    )
    graph = DsaDecodeGraph(layer, fake, cold=cold)
    assert calls.count("qkv") == 3
    assert calls.count("output projection") == 3
    assert calls.count("local experts") == (0 if cold else 3)
    assert calls.count("route") == (3 if cold else 0)
    assert len(graph.after[2]) == (6 if cold else 4)


def test_replay_attends_once_at_dynamic_length_between_fixed_math_graphs():
    calls = []
    state = tuple(torch.ones(1, 2) for _ in range(4))
    before_inputs = tuple(torch.zeros_like(x) for x in state)
    prepared = (*state, *(torch.full((1, 2), n) for n in (2, 3, 4)))
    latent = torch.full((1, 2), 5)
    after_inputs = tuple(torch.zeros_like(x) for x in (latent, *state[1:]))
    outputs = (*state, state[0], state[0])

    def attend(*args):
        assert all(a is b for a, b in zip(args, (prepared[0], *prepared[4:]), strict=True))
        calls.append("append/select/attention")
        return latent

    graph = object.__new__(DsaDecodeGraph)
    graph.layer = SimpleNamespace(dsa_attend=attend)
    graph.before = (SimpleNamespace(replay=lambda: calls.append("before")), before_inputs, prepared)
    graph.after = (SimpleNamespace(replay=lambda: calls.append("after")), after_inputs, outputs)
    assert graph.forward(state) is outputs
    assert calls == ["before", "append/select/attention", "after"]
    assert all(torch.equal(a, b) for a, b in zip(before_inputs, state, strict=True))
    assert torch.equal(after_inputs[0], latent)
    with pytest.raises(ValueError, match="one decode row"):
        graph.forward(tuple(x.expand(2, -1) for x in state))


@pytest.mark.parametrize("changes", [{"linear": True}, {"device": "npu:0"}, {"tp_rank": 0}])
def test_capture_rejects_incompatible_layer_ownership(changes):
    values = dict(linear=False, device="cuda", tp_rank=None)
    values.update(changes)
    with pytest.raises(ValueError, match="unsharded CUDA"):
        DsaDecodeGraph(SimpleNamespace(**values), None, cold=True)
