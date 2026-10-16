# SPDX-License-Identifier: Apache-2.0
"""Candidate dispatch and actual observation seams without device allocation."""

from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.layer_math import TextLayer
from tools.sidecard.layer_weights import NpuExperts
from tools.sidecard.npu_kda import reference
from tools.sidecard.stage_dma import StageDMA


def linear_owner(rows, candidate):
    generator = torch.Generator().manual_seed(624)
    heads, dimension = 8, 128
    owner = object.__new__(TextLayer)
    owner.torch, owner.device, owner.heads = torch, "npu:0", heads
    owner.config = {"rms_norm_eps": 1e-6}
    owner.vectorized_prefill_conv = False
    owner.state = torch.randn(heads, dimension, dimension, generator=generator) / 64
    owner.conv = torch.zeros(3 * heads * dimension, 3).half()
    owner.native_kda = candidate
    owner.weights = {
        "self_attn.conv": torch.randn(3 * heads * dimension, 4, generator=generator).half(),
        "self_attn.A_log": torch.zeros(heads),
        "self_attn.dt_bias": torch.zeros(heads * dimension),
        "self_attn.o_norm.weight": torch.ones(dimension).half(),
    }
    owner.projected = torch.randn(rows, 3 * heads * dimension + heads + 2 * dimension, generator=generator).half()

    def project(value, name):
        if name == "self_attn.in_proj":
            return owner.projected
        if name in ("self_attn.f_b_proj.weight", "self_attn.g_b_proj.weight"):
            return torch.zeros(value.shape[0], heads * dimension).half()
        assert name == "self_attn.o_proj.weight"
        return value

    owner.project = project
    return owner


@pytest.mark.parametrize("rows", [1, 4])
def test_kda_binding_keeps_raw_inputs_fp32_preparation_and_prefill_fallback(rows):
    calls = []

    def native(q, k, value, gate, beta, state):
        calls.append((q.clone(), gate.dtype, beta.dtype, state.dtype))
        return reference(q, k, value, gate, beta, state, torch)

    candidate, baseline = linear_owner(rows, native), linear_owner(rows, None)
    for step in range(3):
        candidate.projected.add_(0.015625 * step)
        baseline.projected.copy_(candidate.projected)
        x = torch.zeros(rows, 4096).half()
        torch.testing.assert_close(candidate.kda(x), baseline.kda(x), rtol=0, atol=0)
        torch.testing.assert_close(candidate.state, baseline.state, rtol=0, atol=0)
        torch.testing.assert_close(candidate.conv, baseline.conv, rtol=0, atol=0)
    assert len(calls) == (3 if rows == 1 else 0)
    for q, gate_dtype, beta_dtype, state_dtype in calls:
        assert q.dtype == torch.float16
        assert gate_dtype == beta_dtype == state_dtype == torch.float32
        assert not torch.allclose(q.float().square().sum(-1), torch.ones(1, 8))


@pytest.mark.parametrize("rows", [1, *range(2, 18), 32])
def test_actual_bank_dispatch_specializes_r1_before_padding_only(rows):
    owner = object.__new__(NpuExperts)
    owner.torch, owner.rank, owner.reduce_results, owner.stable_rows = torch, 0, False, 17
    owner.banks = {18: ()}
    calls = []
    owner.multi_route = lambda layer, x, ids, weights: calls.append(("candidate", x.shape[0])) or x.float()
    owner.native = lambda x, *args: calls.append(("baseline", x.shape[0])) or x.float()
    x = torch.zeros(rows, 4096).half()
    result = owner(18, x, torch.zeros(rows, 8).long(), torch.ones(rows, 8))
    assert result.shape == x.shape
    assert calls == ([("candidate", 1)] if rows == 1 else [("baseline", max(rows, 17))])
    assert owner.stable_rows == 17


def test_router_observer_receives_actual_returned_ids_and_fp32_weights():
    owner = object.__new__(TextLayer)
    owner.torch, owner.device, owner.index = torch, "cpu", 18
    owner.config = {"routed_scaling_factor": 2.5}
    owner.weights = {
        "mlp.gate.weight": torch.arange(80).reshape(10, 8).float() / 80,
        "mlp.gate.e_score_correction_bias": torch.zeros(10),
    }
    seen = []
    owner.execution_diagnostic = SimpleNamespace(route=lambda *args: seen.append(args))
    ids, weights = owner.route(torch.ones(3, 8).half())
    assert seen[0][0] is owner and seen[0][1] is ids and seen[0][2] is weights
    assert ids.dtype == torch.int64 and weights.dtype == torch.float32
    torch.testing.assert_close(weights.sum(-1), torch.full((3,), 2.5))


@pytest.mark.parametrize("field", ["hidden", "ids", "weights"])
def test_malformed_r1_shapes_do_not_reenter_multi_route_fallback(field):
    owner = object.__new__(NpuExperts)
    owner.torch, owner.rank, owner.reduce_results, owner.stable_rows = torch, 0, False, 17
    owner.banks = {18: ()}
    calls = []
    owner.multi_route = lambda *args: pytest.fail("malformed inputs must not recurse through candidate fallback")

    def native(x, *args):
        calls.append(x.shape[0])
        raise ValueError("native shape validation")

    owner.native = native
    fields = {"hidden": torch.zeros(1, 4096).half(), "ids": torch.zeros(1, 8).long(), "weights": torch.ones(1, 8)}
    fields[field] = fields[field][:, :-1]
    with pytest.raises((ValueError, RuntimeError)):
        owner(18, fields["hidden"], fields["ids"], fields["weights"])
    assert calls in ([], [17])


@pytest.mark.parametrize("nonce", ["owner-session", "stale-owner"])
def test_stage_diagnostic_rpc_keeps_mapping_owner_identity(monkeypatch, nonce):
    owner = object.__new__(StageDMA)
    owner.closed, owner.registered, owner.nonce = False, {0, 1}, "owner-session"
    owner.connections = [object(), object()]
    sent = []
    monkeypatch.setattr("tools.sidecard.stage_dma.send_control", lambda connection, packet: sent.append(packet))
    monkeypatch.setattr(
        "tools.sidecard.stage_dma.receive_control",
        lambda connection: {"execution_diagnostic": {"passed": True}, "nonce": nonce},
    )
    if nonce == owner.nonce:
        assert owner.diagnostic_command(1, {"action": "phase"}) == {"passed": True}
    else:
        with pytest.raises(ValueError, match="another stage owner"):
            owner.diagnostic_command(1, {"action": "phase"})
    assert sent == [{"execution_diagnostic": {"action": "phase"}, "nonce": "owner-session"}]
