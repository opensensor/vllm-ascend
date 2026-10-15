# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-math and replay-lifetime CPU gates for partitioned draft MoE."""

from unittest.mock import Mock

import pytest
import torch

from tests.ut.qwen38_1m import test_mtp_six_chip_sharding as fixture_module
from tests.ut.qwen38_1m.test_mtp_six_chip_sharding import (
    configuration,
    policy,
)
from tools.qwen4exp.resident_candidates import draft_graph_experts as candidate


@pytest.fixture
def mtp_classes(monkeypatch):
    return fixture_module.mtp_classes.__wrapped__(monkeypatch)


@pytest.mark.parametrize(
    "ids,offset,count,expected_slots,expected_groups",
    [
        ([4, 2, 4, 3, 2, 8], 2, 3, [1, 4, 3, 0, 2], [(0, 0, 2), (1, 2, 3), (2, 3, 5)]),
        ([0, 1, 8], 2, 3, [], []),
        ([4, 4, 4], 4, 1, [0, 1, 2], [(0, 0, 3)]),
    ],
)
def test_stable_host_routing(ids, offset, count, expected_slots, expected_groups):
    assert candidate.plan_local_routes(ids, offset, count) == (expected_slots, expected_groups)


def bank_and_forward(scope, monkeypatch, capture=None, rank=0, world=1, shared=0):
    torch.manual_seed(941)
    config = configuration(experts=12, shared=shared)
    bank = scope["_MTPFP16MoE"](config, policy(), (rank, world), quantize_experts=True)
    with torch.no_grad():
        bank.gate.normal_(std=0.2)
        for expert in range(bank.num_local_experts):
            bank.load_expert_weight("gate_up_proj", expert, torch.randn(8, 8) * 0.1)
            bank.load_expert_weight("down_proj", expert, torch.randn(8, 4) * 0.1)
        if shared:
            bank.shared_gate_up.normal_(std=0.1)
            bank.shared_down.normal_(std=0.1)
            bank.shared_expert_gate.normal_(std=0.1)
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    forward = candidate.make_forward(
        linear=scope["_linear"],
        route_topk=scope["route_topk"],
        swiglu=scope["swiglu_gate_up"],
        current_capture=lambda: capture,
        weak_ref_tensor=lambda tensor: tensor,
        fallback=fallback,
    )
    return bank, forward, fallback


@pytest.mark.parametrize("tokens", [1, 3, 6, 12])
@pytest.mark.parametrize("rank,world,shared", [(0, 1, 0), (0, 4, 8), (3, 4, 8)])
def test_same_w8a16_route_slot_arithmetic(mtp_classes, monkeypatch, tokens, rank, world, shared):
    bank, forward, fallback = bank_and_forward(mtp_classes, monkeypatch, rank=rank, world=world, shared=shared)
    inputs = torch.randn(tokens, 8, dtype=torch.float16)
    with torch.inference_mode():
        expected = bank._forward_eager(inputs)
        actual = forward(bank, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fallback.assert_not_called()


class Capture:
    _capturing = True

    def __init__(self):
        self.callbacks = []

    def add_eager(self, callback):
        self.callbacks.append(callback)
        self._capturing = False
        try:
            callback()
        finally:
            self._capturing = True


def test_router_and_finish_stay_outside_callback(mtp_classes, monkeypatch):
    capture = Capture()
    bank, forward, _ = bank_and_forward(mtp_classes, monkeypatch, capture=capture, shared=8)
    finish = bank._finish_output
    finish_states = []
    bank._finish_output = lambda x, y: (finish_states.append(capture._capturing), finish(x, y))[1]
    inputs = torch.randn(3, 8, dtype=torch.float16)
    with torch.inference_mode():
        expected = bank._forward_eager(inputs)
        actual = forward(bank, inputs)
    assert len(capture.callbacks) == 1
    assert finish_states == [True, True]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Only the routed callback may run at replay: no router, shared expert,
    # or TP operation is dispatched by that host callback.
    bank._finish_output = Mock(side_effect=AssertionError("finish ran eager"))
    with torch.inference_mode():
        capture.callbacks[0]()


def test_changing_routes_clear_stale_local_slots(mtp_classes, monkeypatch):
    bank, _, _ = bank_and_forward(mtp_classes, monkeypatch, rank=1, world=4)
    inputs = torch.randn(3, 8, dtype=torch.float16)
    output = torch.full((9, 8), float("nan"))
    weights = torch.rand(3, 3)
    pointer = output.data_ptr()
    for ids in [
        torch.tensor([[3, 4, 5], [3, 4, 5], [3, 4, 5]]),
        torch.tensor([[0, 1, 2], [8, 9, 10], [0, 1, 2]]),
        torch.tensor([[0, 3, 2], [5, 9, 10], [0, 4, 2]]),
    ]:
        with torch.inference_mode():
            candidate.fill_routed_output(bank, inputs, weights, ids, output, swiglu=mtp_classes["swiglu_gate_up"])
        assert output.data_ptr() == pointer
        assert torch.isfinite(output).all()
        for slot, expert in enumerate(ids.flatten().tolist()):
            if not bank.expert_offset <= expert < bank.expert_offset + bank.num_local_experts:
                assert torch.count_nonzero(output[slot]) == 0


@pytest.mark.parametrize("case", ["grad", "grouped", "fp16_bank", "large", "empty", "dtype", "cpu"])
def test_unsupported_modes_use_original_forward(mtp_classes, monkeypatch, case):
    bank, forward, fallback = bank_and_forward(mtp_classes, monkeypatch)
    fallback.side_effect = None
    fallback.return_value = object()
    inputs = torch.randn(3, 8, dtype=torch.float16)
    if case == "grouped":
        bank.grouped_experts = True
    elif case == "fp16_bank":
        bank.quantized_experts = False
    elif case == "large":
        inputs = torch.randn(candidate.MAX_TOKENS + 1, 8, dtype=torch.float16)
    elif case == "empty":
        inputs = inputs[:0]
    elif case == "dtype":
        inputs = inputs.float()
    elif case == "cpu":
        monkeypatch.setattr(candidate, "_supports_device", lambda device: False)
    with torch.set_grad_enabled(case == "grad"):
        assert forward(bank, inputs) is fallback.return_value
    fallback.assert_called_once_with(bank, inputs)
