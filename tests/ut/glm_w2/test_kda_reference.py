# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the eager KDA parity oracle (tools/glm_w2/kda_reference.py).

Contracts covered (plan task G-ref):
  * shape/dtype (H=64, head_dim=128, arbitrary seq len; fp16-castable output),
  * strict causality (a perturbation at t' > t cannot change output at t),
  * chunked == recurrent (state-carry consistency oracle for G4),
  * a hand-computed 2-step recurrence with a known decay,
  * gate bounds + monotonicity, short-conv causality + kernel width,
  * initial-state carry equivalence, determinism.

Run: python3 -m pytest -q --noconftest tests/ut/glm_w2/test_kda_reference.py
"""

import os
import sys

import pytest
import torch

# tools/ is not an installed package; add repo root so the import resolves
# whether pytest is invoked from the repo root or elsewhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.glm_w2.kda_reference import (  # noqa: E402
    GLM_HEAD_DIM,
    GLM_NUM_HEADS,
    KDA_GATE_LOWER_BOUND,
    gated_rmsnorm,
    kda_chunked_reference,
    kda_layer_reference,
    kda_recurrent_reference,
    kda_safe_gate,
    short_conv1d_causal,
)

torch.manual_seed(0)

H = GLM_NUM_HEADS      # 64
D = GLM_HEAD_DIM       # 128


def _rand_qkvg(T, h=H, d=D, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(T, h, d, generator=gen)
    k = torch.randn(T, h, d, generator=gen)
    v = torch.randn(T, h, d, generator=gen)
    # log-decay gate in (lower_bound, 0): use the real safe gate for realism.
    raw_g = torch.randn(T, h, d, generator=gen)
    a_log = torch.randn(h, generator=gen)
    g = kda_safe_gate(raw_g, a_log, None)
    beta = torch.randn(T, h, generator=gen)  # raw (pre-sigmoid)
    return q, k, v, g, beta


# --------------------------------------------------------------------------- #
# shape / dtype contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [1, 5, 37, 128])
def test_shape_and_dtype_contract(T):
    q, k, v, g, beta = _rand_qkvg(T)
    out, state = kda_recurrent_reference(q, k, v, g, beta)
    assert out.shape == (T, H, D)
    assert state.shape == (H, D, D)
    assert out.dtype == torch.float32
    # fp16-castable, finite.
    half = out.half()
    assert torch.isfinite(half).all()


def test_batch_dim_of_one_accepted():
    T = 8
    q, k, v, g, beta = _rand_qkvg(T)
    out4, _ = kda_recurrent_reference(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0)
    )
    out3, _ = kda_recurrent_reference(q, k, v, g, beta)
    assert out4.shape == (1, T, H, D)
    torch.testing.assert_close(out4[0], out3)


def test_batch_size_gt_one_rejected():
    q, k, v, g, beta = _rand_qkvg(4)
    with pytest.raises(ValueError):
        kda_recurrent_reference(
            q.unsqueeze(0).repeat(2, 1, 1, 1),
            k.unsqueeze(0).repeat(2, 1, 1, 1),
            v.unsqueeze(0).repeat(2, 1, 1, 1),
            g.unsqueeze(0).repeat(2, 1, 1, 1),
            beta.unsqueeze(0).repeat(2, 1, 1),
        )


# --------------------------------------------------------------------------- #
# causality: output at t depends only on inputs <= t
# --------------------------------------------------------------------------- #
def test_recurrence_causality():
    T = 24
    q, k, v, g, beta = _rand_qkvg(T, seed=1)
    out_ref, _ = kda_recurrent_reference(q, k, v, g, beta)

    t_cut = 10  # perturb everything strictly after t_cut
    gen = torch.Generator().manual_seed(99)
    q2, k2, v2, beta2 = q.clone(), k.clone(), v.clone(), beta.clone()
    q2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, D, generator=gen)
    k2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, D, generator=gen)
    v2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, D, generator=gen)
    beta2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, generator=gen)
    out_pert, _ = kda_recurrent_reference(q2, k2, v2, g, beta2)

    # outputs at <= t_cut must be bit-identical; at > t_cut may differ.
    torch.testing.assert_close(out_pert[: t_cut + 1], out_ref[: t_cut + 1])
    assert not torch.allclose(out_pert[t_cut + 1:], out_ref[t_cut + 1:])


def test_short_conv_causality_and_width():
    T = 20
    C = 6
    width = 4
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(T, C, generator=gen)
    w = torch.randn(C, width, generator=gen)
    out_ref = short_conv1d_causal(x, w, activation=None)

    # Perturbing x at t' > t must not change conv output at t.
    t_cut = 9
    x2 = x.clone()
    x2[t_cut + 1:] += torch.randn(T - t_cut - 1, C, generator=gen)
    out_pert = short_conv1d_causal(x2, w, activation=None)
    torch.testing.assert_close(out_pert[: t_cut + 1], out_ref[: t_cut + 1])

    # Kernel width: with only the last tap set (delta at t), conv == identity.
    w_id = torch.zeros(C, width)
    w_id[:, -1] = 1.0
    out_id = short_conv1d_causal(x, w_id, activation=None)
    torch.testing.assert_close(out_id, x, atol=1e-5, rtol=1e-5)

    # And the receptive field is exactly `width`: only the last tap set at
    # offset (width-1) back reproduces a shifted input.
    w_shift = torch.zeros(C, width)
    w_shift[:, 0] = 1.0  # tap furthest in the past -> shift by width-1
    out_shift = short_conv1d_causal(x, w_shift, activation=None)
    torch.testing.assert_close(out_shift[width - 1:], x[: T - (width - 1)], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_shift[: width - 1], torch.zeros(width - 1, C), atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# chunked == recurrent (state-carry consistency)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", [1, 4, 16, 64])
def test_chunked_equals_recurrent(chunk):
    T = 70
    q, k, v, g, beta = _rand_qkvg(T, seed=3)
    out_full, st_full = kda_recurrent_reference(q, k, v, g, beta)
    out_chunk, st_chunk = kda_chunked_reference(q, k, v, g, beta, chunk_size=chunk)
    torch.testing.assert_close(out_chunk, out_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_chunk, st_full, atol=1e-5, rtol=1e-5)


def test_initial_state_carry_equivalence():
    T = 30
    split = 11
    q, k, v, g, beta = _rand_qkvg(T, seed=4)
    out_full, st_full = kda_recurrent_reference(q, k, v, g, beta)

    out_a, st_a = kda_recurrent_reference(
        q[:split], k[:split], v[:split], g[:split], beta[:split]
    )
    out_b, st_b = kda_recurrent_reference(
        q[split:], k[split:], v[split:], g[split:], beta[split:], initial_state=st_a
    )
    torch.testing.assert_close(torch.cat([out_a, out_b], 0), out_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_b, st_full, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# hand-computed 2-step recurrence (known decay)
# --------------------------------------------------------------------------- #
def test_hand_computed_two_step_with_decay():
    # H=1, K=V=2, no l2norm, scale=1, beta=1, decay exp(g)=0.5 (g=ln 0.5).
    # k0=[1,0] q0=[1,0] v0=[2,0]; k1=[1,0] q1=[1,0] v1=[0,3].
    # Worked out by hand (see kda_reference docstring math):
    #   t0: S = outer(v0,k0)=[[2,0],[0,0]];         o0 = S@q0 = [2,0]
    #   t1: S*=0.5 -> [[1,0],[0,0]]; Sk=[1,0]; u=v1-Sk=[-1,3];
    #       S += outer(u,k1) = [[0,0],[3,0]];        o1 = S@q1 = [0,3]
    q = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])   # [T=2, H=1, K=2]
    k = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    v = torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]]])
    ln_half = torch.log(torch.tensor(0.5))
    g = torch.full((2, 1, 2), float(ln_half))
    beta = torch.ones(2, 1)  # already-sigmoided value 1.0

    out, state = kda_recurrent_reference(
        q, k, v, g, beta, scale=1.0, use_qk_l2norm=False, beta_is_raw=False
    )
    expected = torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    # final state = [[0,0],[3,0]]
    torch.testing.assert_close(state[0], torch.tensor([[0.0, 0.0], [3.0, 0.0]]), atol=1e-6, rtol=1e-6)


def test_zero_input_gives_zero_output():
    T = 6
    q = torch.randn(T, H, D)
    k = torch.randn(T, H, D)
    v = torch.zeros(T, H, D)
    g = kda_safe_gate(torch.randn(T, H, D), torch.randn(H), None)
    beta = torch.randn(T, H)
    out, state = kda_recurrent_reference(q, k, v, g, beta)
    # v==0 -> delta update u = -S@k*beta but S starts 0 and only grows via u*k;
    # with v identically 0 the state stays 0 forever, so output is 0.
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state, torch.zeros_like(state), atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# gate: bounds + monotonicity
# --------------------------------------------------------------------------- #
def test_safe_gate_bounds():
    raw_g = torch.randn(12, H, D) * 10.0  # wide range
    a_log = torch.randn(H)
    g = kda_safe_gate(raw_g, a_log, None, lower_bound=KDA_GATE_LOWER_BOUND)
    # log-decay in [lower_bound, 0]: mathematically open, but fp32 sigmoid
    # saturates to exactly 0.0 or 1.0 at the extremes, so the realized bounds
    # are inclusive. exp(g) is then a valid decay in [e^lb, 1].
    assert (g <= 0).all()
    assert (g >= KDA_GATE_LOWER_BOUND).all()
    decay = torch.exp(g)
    assert (decay > 0).all()
    assert (decay <= 1).all()


def test_safe_gate_monotonic_in_raw():
    # larger raw_g -> larger sigmoid -> more-negative gate -> stronger forgetting.
    a_log = torch.zeros(1)  # exp(a)=1
    lo = kda_safe_gate(torch.full((1, 1, 1), -3.0), a_log, None)
    hi = kda_safe_gate(torch.full((1, 1, 1), 3.0), a_log, None)
    assert hi.item() < lo.item()          # more negative
    assert torch.exp(hi).item() < torch.exp(lo).item()  # more decay


def test_dt_bias_shifts_gate():
    raw_g = torch.zeros(1, 2, 3)
    a_log = torch.zeros(2)
    no_bias = kda_safe_gate(raw_g, a_log, None)
    bias = torch.full((2 * 3,), 5.0)
    with_bias = kda_safe_gate(raw_g, a_log, bias)
    assert (with_bias < no_bias).all()  # positive bias -> stronger decay


def test_gate_head_mismatch_raises():
    with pytest.raises(ValueError):
        kda_safe_gate(torch.randn(4, 8, D), torch.randn(H), None)


# --------------------------------------------------------------------------- #
# output gated RMSNorm
# --------------------------------------------------------------------------- #
def test_gated_rmsnorm_matches_manual():
    x = torch.randn(5, H, D)
    g = torch.randn(5, H, D)
    w = torch.randn(D)
    out = gated_rmsnorm(x, g, w)
    # manual
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    manual = xf * torch.rsqrt(var + 1e-5) * w.float() * torch.sigmoid(g.float())
    torch.testing.assert_close(out, manual)
    # RMS of the (pre-gate, unit-weight) normalized tensor is ~1.
    normed = xf * torch.rsqrt(var + 1e-5)
    rms = normed.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
# determinism + end-to-end layer wiring
# --------------------------------------------------------------------------- #
def test_determinism():
    q, k, v, g, beta = _rand_qkvg(15, seed=7)
    o1, s1 = kda_recurrent_reference(q, k, v, g, beta)
    o2, s2 = kda_recurrent_reference(q, k, v, g, beta)
    torch.testing.assert_close(o1, o2, atol=0, rtol=0)
    torch.testing.assert_close(s1, s2, atol=0, rtol=0)


def test_layer_reference_shape_and_causality():
    T = 16
    gen = torch.Generator().manual_seed(8)
    proj = H * D
    qkv = torch.randn(T, 3 * proj, generator=gen)
    conv_w = torch.randn(3 * proj, 4, generator=gen)
    raw_g = torch.randn(T, proj, generator=gen)
    beta_raw = torch.randn(T, H, generator=gen)
    a_log = torch.randn(H, generator=gen)
    dt_bias = torch.randn(proj, generator=gen)
    g_out = torch.randn(T, proj, generator=gen)
    o_w = torch.randn(D, generator=gen)

    out = kda_layer_reference(
        qkv, conv_weight=conv_w, raw_g=raw_g, beta_raw=beta_raw,
        a_log=a_log, dt_bias=dt_bias, g_out=g_out, o_norm_weight=o_w,
    )
    assert out.shape == (T, H, D)
    assert torch.isfinite(out.half()).all()

    # causality end-to-end: perturbing a late token leaves early outputs intact
    # up to the conv receptive field (width-1=3 lookahead is impossible; conv is
    # causal), so outputs strictly before the perturbation are unchanged.
    t_cut = 8
    qkv2 = qkv.clone()
    qkv2[t_cut + 1:] += torch.randn(T - t_cut - 1, 3 * proj, generator=gen)
    out2 = kda_layer_reference(
        qkv2, conv_weight=conv_w, raw_g=raw_g, beta_raw=beta_raw,
        a_log=a_log, dt_bias=dt_bias, g_out=g_out, o_norm_weight=o_w,
    )
    torch.testing.assert_close(out2[: t_cut + 1], out[: t_cut + 1], atol=1e-5, rtol=1e-5)
