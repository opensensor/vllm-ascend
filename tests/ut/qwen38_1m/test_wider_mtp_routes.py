# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import wider_mtp_routes as candidate


@pytest.mark.parametrize("tokens", [4, 6, 12])
def test_gate_schedule_admits_only_matching_routed_input_broadcast(tokens):
    prepared = (SimpleNamespace(shape=(tokens, 1280), device=SimpleNamespace(type="npu")),)
    routes = torch.arange(tokens * 10, dtype=torch.int32)
    bank = SimpleNamespace(
        backend=candidate.NATIVE_INT4_BACKEND,
        weight=SimpleNamespace(ndim=3, shape=(86, 1280, 1280)),
        weight_scale=object(),
        weight_offset=object(),
        weight_sum=object(),
    )
    marker = object()
    calls = []

    def gate(*args):
        calls.append(args)
        return marker

    linear = candidate.make_gate_linear(lambda *args: "fallback", gate)
    assert linear(bank, prepared, routes) is marker
    assert calls[0][0] is prepared[0] and calls[0][-1] is routes
    assert linear(bank, prepared, routes.to(torch.int64)) == "fallback"
    prepared[0].shape = (tokens * 10, 1280)
    assert linear(bank, prepared, routes) == "fallback"


@pytest.mark.parametrize("tokens", [4, 6])
def test_wider_routed_preserves_expert_offset_and_weight_buffers(monkeypatch, tokens):
    inputs = SimpleNamespace(shape=(tokens, 2560), device=SimpleNamespace(type="npu"))
    ids = torch.arange(tokens * 10).reshape(tokens, 10) + 86
    weights = torch.rand(tokens, 10)
    bank = SimpleNamespace(weight=object(), weight_scale=object(), weight_offset=object(), weight_sum=object())
    calls = []

    def gate(prepared, routes):
        assert prepared == ("input",)
        torch.testing.assert_close(routes, torch.arange(tokens * 10, dtype=torch.int32))
        return "projected"

    monkeypatch.setattr(candidate, "pack_activation_device", lambda _: ("input",))
    monkeypatch.setattr(candidate, "swiglu_pack_activation_device", lambda value: (value,))
    module = SimpleNamespace(
        native_int4=True,
        fused_native_down_reduce=True,
        top_k=10,
        expert_offset=86,
        projections={"gate_up_proj": SimpleNamespace(native_linear=gate), "down_proj": bank},
    )
    output = object()

    def down(*args):
        calls.append(args)
        return output

    assert candidate.make_routed(None, down)(module, inputs, weights, ids) is output
    assert calls[0][:5] == ("projected", bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum)
    assert calls[0][-1] is weights


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 7, 12, 24, 640])
def test_unqualified_token_shapes_keep_original_dispatch(tokens):
    module = SimpleNamespace(native_int4=True, fused_native_down_reduce=True)
    inputs = SimpleNamespace(shape=(tokens, 2560), device=SimpleNamespace(type="npu"))
    marker = object()
    assert candidate.make_routed(lambda *args: marker, None)(module, inputs, None, None) is marker
