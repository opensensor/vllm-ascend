# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline coverage for the no-reload resident real-weight numerical gate."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.qwen38_1m import test_mtp_six_chip_sharding as fixture_module
from tests.ut.qwen38_1m.test_draft_graph_experts_candidate import bank_and_forward
from tools.qwen4exp import draft_real_weight_gate as gate


@pytest.fixture
def mtp_classes(monkeypatch):
    return fixture_module.mtp_classes.__wrapped__(monkeypatch)


@pytest.mark.parametrize("offset", [0, 128, 256, 384])
@pytest.mark.parametrize("tokens", [1, 3, 6])
def test_forced_routes_are_valid_and_cover_local_boundaries(offset, tokens):
    cases = gate.forced_route_ids(offset, 128, 512, 10, tokens)
    for name, ids in cases.items():
        assert ids.shape == (tokens, 10)
        assert torch.all((ids >= 0) & (ids < 512))
        assert all(len(set(row)) == 10 for row in ids.tolist())
        mask = (ids >= offset) & (ids < offset + 128)
        if name.startswith("remote"):
            assert not mask.any()
        elif name.startswith("local"):
            assert mask.all()
        else:
            assert mask.any() and not mask.all()
            assert offset in ids[0] and offset + 127 in ids[0]
            assert (offset - 1) % 512 in ids[0] and (offset + 128) % 512 in ids[0]


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("tokens", [1, 3, 6])
def test_forced_route_reference_matches_original_model_algorithm(mtp_classes, monkeypatch, rank, tokens):
    bank, _, _ = bank_and_forward(mtp_classes, monkeypatch, rank=rank, world=4, shared=8)
    inputs = torch.randn(tokens, 8, dtype=torch.float16)
    logits = mtp_classes["_linear"](inputs, bank.gate, bank.policy.router_dtype)
    weights, ids = mtp_classes["route_topk"](logits, bank.top_k)
    with torch.inference_mode():
        expected = bank._forward_eager(inputs)
        actual = gate.reference_routed_output(
            bank,
            inputs,
            weights,
            ids,
            select_local_routes=mtp_classes["_select_local_mtp_routes"],
            swiglu=mtp_classes["swiglu_gate_up"],
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_view_shares_weights_without_mutating_collective_fields(mtp_classes, monkeypatch):
    bank, _, _ = bank_and_forward(mtp_classes, monkeypatch, rank=2, world=4)
    reduction = Mock(side_effect=AssertionError("network collective must not run"))
    bank._tp_reduce = reduction
    versions = gate.parameter_versions(bank)
    view = gate.local_view(bank)
    assert view is not bank
    assert view.expert_offset == bank.expert_offset
    assert view.expert_tp_size == 1 and view._tp_reduce is None
    assert bank.expert_tp_size == 4 and bank._tp_reduce is reduction
    assert view.gate_up_proj is bank.gate_up_proj
    assert view.down_proj is bank.down_proj
    with torch.inference_mode():
        view._finish_output(torch.zeros(3, 8).half(), torch.zeros(3, 8))
    assert gate.parameter_versions(bank) == versions
    reduction.assert_not_called()


def test_versions_support_inference_created_parameters():
    bank = torch.nn.Linear(3, 2)
    with torch.inference_mode():
        bank.weight = torch.nn.Parameter(torch.ones(2, 3))
    versions = gate.parameter_versions(bank)
    assert next(version for name, _, version in versions if name == "weight") is None


def test_find_loaded_draft_bank_through_wrappers_and_aliases(mtp_classes, monkeypatch):
    bank, _, _ = bank_and_forward(mtp_classes, monkeypatch)
    model = torch.nn.Module()
    model.first = bank
    model.alias = bank
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(drafter=SimpleNamespace(model=SimpleNamespace(runnable=model)))
    )
    banks = gate.find_draft_banks(worker)
    assert len(banks) == 1 and banks[0][1] is bank


def test_missing_draft_fails_before_any_device_work():
    with pytest.raises(RuntimeError, match="no loaded draft"):
        gate.find_draft_banks(SimpleNamespace(model_runner=SimpleNamespace()))


def test_qualified_geometry_rejects_small_synthetic_bank(mtp_classes, monkeypatch):
    bank, _, _ = bank_and_forward(mtp_classes, monkeypatch)
    with pytest.raises(ValueError, match="qualified routed W8A16 geometry"):
        gate.validate_bank(bank)


def test_rpc_uses_loaded_bank_and_preserves_rank(monkeypatch):
    worker = gate.QfnDraftRealWeightMixin()
    worker.rank = 3
    bank = object()
    monkeypatch.setattr(gate, "find_draft_banks", lambda owner: [("drafter.model.mlp", bank)])
    runner = Mock(return_value={"comparisons": 60})
    monkeypatch.setattr(gate, "gate_bank", runner)
    result = worker.qfn_draft_real_weight_gate()
    assert result == {"rank": 3, "passed": True, "banks": [{"name": "drafter.model.mlp", "comparisons": 60}]}
    runner.assert_called_once_with(bank, tokens=(1, 3, 6), repeats=2, seed=9401)


@pytest.mark.parametrize("tokens,repeats", [([], 1), ([0], 1), ([13], 1), ([True], 1), ([3], 0), ([3], True)])
def test_rpc_rejects_invalid_requests_before_bank_lookup(monkeypatch, tokens, repeats):
    lookup = Mock(side_effect=AssertionError("unexpected model access"))
    monkeypatch.setattr(gate, "find_draft_banks", lookup)
    with pytest.raises(ValueError):
        gate.QfnDraftRealWeightMixin().qfn_draft_real_weight_gate(tokens=tokens, repeats=repeats)
    lookup.assert_not_called()
