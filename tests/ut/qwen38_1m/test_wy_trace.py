# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
import torch

from tests.ut.qwen38_1m.reference.gdn_reference import gdn_delta_rule_recurrent
from tools.qwen4exp.wy_trace import capture_wy_trace, wy_downstream_fp64, wy_fp64


def inputs(tokens=64, kh=2, vh=6):
    gen = torch.Generator().manual_seed(tokens + vh)
    q = torch.nn.functional.normalize(torch.randn(1, tokens, kh, 4, generator=gen), dim=-1).half()
    k = torch.nn.functional.normalize(torch.randn(1, tokens, kh, 4, generator=gen), dim=-1).half()
    v = torch.randn(1, tokens, vh, 4, generator=gen).half()
    g = -torch.rand(1, tokens, vh, generator=gen) * 0.1
    beta = torch.rand(1, tokens, vh, generator=gen).half()
    return q, k, v, g, beta


@pytest.mark.parametrize("tokens,kh,vh", [(64, 2, 6), (128, 4, 12), (64, 16, 48)])
def test_fp64_oracle_matches_independent_unit_lower_solve(tokens, kh, vh):
    q, k, v, g, beta = inputs(tokens, kh, vh)
    _, _, w, u, cumulative = wy_fp64(q, k, v, g, beta)
    key = k.double().repeat_interleave(vh // kh, 2).transpose(1, 2).reshape(1, vh, tokens // 64, 64, 4)
    value = v.double().transpose(1, 2).reshape_as(key)
    b = beta.double().transpose(1, 2).reshape(1, vh, tokens // 64, 64)
    gates = g.double().transpose(1, 2).reshape_as(b).cumsum(-1)
    lower = torch.tril(
        (key @ key.transpose(-1, -2)) * b.unsqueeze(-1) * (gates.unsqueeze(-1) - gates.unsqueeze(-2)).exp(), diagonal=-1
    )
    unit = torch.eye(64, dtype=torch.float64) + lower
    expected_u = torch.linalg.solve_triangular(unit, value * b.unsqueeze(-1), upper=False)
    expected_w = torch.linalg.solve_triangular(unit, key * (b * gates.exp()).unsqueeze(-1), upper=False)
    torch.testing.assert_close(u, expected_u.reshape_as(u), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(w, expected_w.reshape_as(w), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cumulative, gates.reshape_as(cumulative), rtol=0, atol=0)


def test_failure_saves_outputs_and_both_final_states_after_downstream_mismatch(tmp_path):
    values = inputs()
    state = torch.ones(1, 6, 4, 4)
    original = state.clone()

    def prepare(*args):
        return tuple(t.float() for t in wy_fp64(*args))

    def bad_prepare(*args):
        result = list(prepare(*args))
        result[3] = result[3] + 0.1
        return tuple(result)

    def downstream(prepared, local_state):
        local_state.add_(2)
        return prepared[3], local_state

    directory = tmp_path / "trace"
    result = capture_wy_trace(
        directory,
        values,
        state,
        reference_prepare=prepare,
        candidate_prepare=bad_prepare,
        downstream=downstream,
        qk_normalized=True,
    )
    assert not result["passed"] and result["first_divergence"] == "u"
    assert not result["comparisons"]["output"]["passed"]
    assert result["comparisons"]["final_state"]["passed"]
    archive = np.load(directory / "tensors.npz", allow_pickle=False)
    np.testing.assert_array_equal(archive["initial_state"], original.numpy())
    assert "candidate_final_state" in archive and "fp64_w" in archive
    assert torch.equal(state, original)
    assert json.loads((directory / "trace.json").read_text())["archive_sha256"] == result["archive_sha256"]


def test_exception_and_input_mutation_reject_but_preserve_inputs(tmp_path):
    values = inputs()

    def broken(*args):
        args[0].zero_()
        raise RuntimeError("diagnostic failure")

    result = capture_wy_trace(
        tmp_path / "failed",
        values,
        torch.zeros(1, 6, 4, 4),
        reference_prepare=broken,
        candidate_prepare=broken,
        downstream=lambda *_: None,
        qk_normalized=True,
    )
    assert not result["passed"] and result["input_mutation"] == {"reference": True, "candidate": True}
    assert set(result["errors"]) == {"reference", "candidate"}
    assert not result["comparisons"]


@pytest.mark.parametrize("tokens", [64, 128])
def test_complete_chunk_outputs_and_state_match_independent_token_recurrence(tokens):
    q, k, v, g, beta = inputs(tokens)
    initial = torch.arange(6 * 4 * 4, dtype=torch.float64).reshape(1, 6, 4, 4) * 0.001
    output, state = wy_downstream_fp64(wy_fp64(q, k, v, g, beta), initial)
    expected, final = gdn_delta_rule_recurrent(
        q[0].repeat_interleave(3, 1), k[0].repeat_interleave(3, 1), v[0], g[0], beta[0], initial[0]
    )
    torch.testing.assert_close(output[0].transpose(0, 1), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(state[0], final, rtol=1e-10, atol=1e-10)


def test_zero_beta_and_extreme_negative_gates_remain_finite():
    values = list(inputs(128))
    values[-1].zero_()
    values[-2].fill_(-100.0)
    prepared = wy_fp64(*values)
    assert torch.count_nonzero(prepared[2]) == torch.count_nonzero(prepared[3]) == 0
    assert all(torch.isfinite(t).all() for t in prepared)
