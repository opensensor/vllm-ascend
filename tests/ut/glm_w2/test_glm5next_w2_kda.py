# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the Triton-free 310P KDA module (plan task G4).

The authoritative correctness contract is **parity against the G-ref eager
oracle** in ``tools/glm_w2/kda_reference.py`` (task G-ref). The 310P module
``vllm_ascend/models/glm5next_w2/kda.py`` must reproduce the exact
Kimi-Delta-Attention gated-delta-rule math of the shipped 910/Triton path --
short depthwise causal conv1d + SiLU, bounded "safe" sigmoid decay gate,
per-head-scalar-beta gated delta-rule recurrence with per-K-channel decay, and
a sigmoid-gated output RMSNorm -- with NO Triton and NO ``vllm_ascend.ops.triton``
on the import path, and runnable on a plain-``torch`` CPU host (``torch_npu``
absent) so this test can execute at all.

Tolerances
----------
* fp32 recurrence parity: the module contracts the recurrent state with batched
  ``torch.bmm`` (NPU-friendly) where the oracle uses an elementwise
  multiply-then-``sum(-1)``. The two are the same math but reassociate the
  K-reduction, so fp32 rounding differs at the ~1e-6 level and can accrete over
  the sequence through the carried state. We assert ``atol=1e-4, rtol=1e-4``
  (measured worst-case over the tested seq lens is well under this; see
  ``test_recurrence_parity_measured_drift``).
* fp16 end-to-end (policy cast site ``kda`` -> float16): the core output is cast
  to float16 at the layer boundary, so ~1e-2 absolute is expected; we assert
  ``atol=3e-2, rtol=3e-2`` against the fp32 oracle layer output.

Run: python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm5next_w2_kda.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# tools/ is not an installed package; add repo root so the oracle import resolves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.glm_w2.kda_reference import (  # noqa: E402
    GLM_HEAD_DIM,
    GLM_NUM_HEADS,
    KDA_GATE_LOWER_BOUND,
    kda_layer_reference,
    kda_recurrent_reference,
    kda_safe_gate,
)

# The unit under test: the Triton-free 310P KDA module. Imported both directly
# and (below) lazily from the package to prove the export + import hygiene.
from vllm_ascend.models.glm5next_w2.kda import Glm5NextW2KDA  # noqa: E402

torch.manual_seed(0)

H = GLM_NUM_HEADS      # 64
D = GLM_HEAD_DIM       # 128
_KDA_SRC = Path(_REPO_ROOT) / "vllm_ascend" / "models" / "glm5next_w2" / "kda.py"

# fp32 recurrence parity tolerance (bmm vs sum reassociation; see module docstring).
FP32_ATOL = 1e-4
FP32_RTOL = 1e-4
# fp16 boundary-cast tolerance for the end-to-end layer.
FP16_ATOL = 3e-2
FP16_RTOL = 3e-2

# Sequence lengths exercised: single-token decode, small prefill, an exact
# chunk boundary (64), and a chunk boundary + 1 to catch off-by-one state carry.
SEQ_LENS = [1, 5, 64, 65]


def _rand_qkvg(T, h=H, d=D, seed=0):
    """Random q/k/v, a realistic log-decay gate (via the safe gate), raw beta."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(T, h, d, generator=gen)
    k = torch.randn(T, h, d, generator=gen)
    v = torch.randn(T, h, d, generator=gen)
    raw_g = torch.randn(T, h, d, generator=gen)
    a_log = torch.randn(h, generator=gen)
    g = kda_safe_gate(raw_g, a_log, None)  # [T, H, D] log-decay in (lb, 0)
    beta = torch.randn(T, h, generator=gen)  # raw (pre-sigmoid)
    return q, k, v, g, beta


def _module():
    return Glm5NextW2KDA(num_heads=H, head_dim=D)


# --------------------------------------------------------------------------- #
# PRIMARY CONTRACT: recurrence parity vs the G-ref oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", SEQ_LENS)
def test_recurrence_parity_vs_oracle(T):
    q, k, v, g, beta = _rand_qkvg(T, seed=T)
    mod = _module()
    out_mod, st_mod = mod.recurrence(q, k, v, g, beta)
    out_ref, st_ref = kda_recurrent_reference(q, k, v, g, beta)

    assert out_mod.shape == (T, H, D)
    assert st_mod.shape == (H, D, D)
    assert out_mod.dtype == torch.float32
    torch.testing.assert_close(out_mod, out_ref, atol=FP32_ATOL, rtol=FP32_RTOL)
    torch.testing.assert_close(st_mod, st_ref, atol=FP32_ATOL, rtol=FP32_RTOL)


def test_recurrence_parity_measured_drift():
    """Report the actual worst-case drift so the chosen tolerance is honest."""
    worst = 0.0
    for T in SEQ_LENS:
        q, k, v, g, beta = _rand_qkvg(T, seed=100 + T)
        out_mod, _ = _module().recurrence(q, k, v, g, beta)
        out_ref, _ = kda_recurrent_reference(q, k, v, g, beta)
        worst = max(worst, (out_mod - out_ref).abs().max().item())
    # Comfortably inside FP32_ATOL; if this ever regresses the tol is a lie.
    assert worst < FP32_ATOL, f"worst-case recurrence drift {worst} exceeds {FP32_ATOL}"


def test_single_token_decode_step():
    """T=1 (decode) must match and carry a correct one-step state."""
    q, k, v, g, beta = _rand_qkvg(1, seed=7)
    out_mod, st_mod = _module().recurrence(q, k, v, g, beta)
    out_ref, st_ref = kda_recurrent_reference(q, k, v, g, beta)
    torch.testing.assert_close(out_mod, out_ref, atol=FP32_ATOL, rtol=FP32_RTOL)
    torch.testing.assert_close(st_mod, st_ref, atol=FP32_ATOL, rtol=FP32_RTOL)


# --------------------------------------------------------------------------- #
# initial-state carry + chunked == whole-sequence (state threading for prefill)
# --------------------------------------------------------------------------- #
def test_initial_state_carry_matches_oracle():
    T, split = 40, 17
    q, k, v, g, beta = _rand_qkvg(T, seed=5)
    mod = _module()
    out_a, st_a = mod.recurrence(q[:split], k[:split], v[:split], g[:split], beta[:split])
    out_b, st_b = mod.recurrence(
        q[split:], k[split:], v[split:], g[split:], beta[split:], initial_state=st_a
    )
    out_ref, st_ref = kda_recurrent_reference(q, k, v, g, beta)
    torch.testing.assert_close(torch.cat([out_a, out_b], 0), out_ref, atol=FP32_ATOL, rtol=FP32_RTOL)
    torch.testing.assert_close(st_b, st_ref, atol=FP32_ATOL, rtol=FP32_RTOL)


@pytest.mark.parametrize("chunk", [1, 16, 64])
def test_chunked_equals_recurrent(chunk):
    T = 70
    q, k, v, g, beta = _rand_qkvg(T, seed=3)
    mod = _module()
    out_full, st_full = mod.recurrence(q, k, v, g, beta)
    out_chunk, st_chunk = mod.chunked_recurrence(q, k, v, g, beta, chunk_size=chunk)
    torch.testing.assert_close(out_chunk, out_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_chunk, st_full, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# causality: output at t depends only on inputs <= t
# --------------------------------------------------------------------------- #
def test_recurrence_causality():
    T = 24
    q, k, v, g, beta = _rand_qkvg(T, seed=1)
    mod = _module()
    out_ref, _ = mod.recurrence(q, k, v, g, beta)
    t_cut = 10
    gen = torch.Generator().manual_seed(99)
    q2, k2, v2, beta2 = q.clone(), k.clone(), v.clone(), beta.clone()
    q2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, D, generator=gen)
    k2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, D, generator=gen)
    v2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, D, generator=gen)
    beta2[t_cut + 1:] += torch.randn(T - t_cut - 1, H, generator=gen)
    out_pert, _ = mod.recurrence(q2, k2, v2, g, beta2)
    torch.testing.assert_close(out_pert[: t_cut + 1], out_ref[: t_cut + 1], atol=FP32_ATOL, rtol=FP32_RTOL)
    assert not torch.allclose(out_pert[t_cut + 1:], out_ref[t_cut + 1:])


# --------------------------------------------------------------------------- #
# short conv: width sanity + causality (matches the oracle's conv contract)
# --------------------------------------------------------------------------- #
def test_short_conv_width_and_causality():
    T, C, width = 20, 6, 4
    mod = _module()
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(T, C, generator=gen)

    # Delta kernel on the last (current) tap => identity (causal, no lookahead).
    w_id = torch.zeros(C, width)
    w_id[:, -1] = 1.0
    out_id = mod.short_conv1d(x, w_id, activation=None)
    torch.testing.assert_close(out_id, x, atol=1e-5, rtol=1e-5)

    # Kernel width exactly `width`: furthest-past tap shifts input by width-1.
    w_shift = torch.zeros(C, width)
    w_shift[:, 0] = 1.0
    out_shift = mod.short_conv1d(x, w_shift, activation=None)
    torch.testing.assert_close(out_shift[width - 1:], x[: T - (width - 1)], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_shift[: width - 1], torch.zeros(width - 1, C), atol=1e-5, rtol=1e-5)

    # Causality: perturbing a late token cannot change an earlier conv output.
    t_cut = 9
    x2 = x.clone()
    x2[t_cut + 1:] += torch.randn(T - t_cut - 1, C, generator=gen)
    out_a = mod.short_conv1d(x, w_shift)
    out_b = mod.short_conv1d(x2, w_shift)
    torch.testing.assert_close(out_b[: t_cut + 1], out_a[: t_cut + 1], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# gate: bounded decay in [e^-5, 1] and log-decay in [lower_bound, 0]
# --------------------------------------------------------------------------- #
def test_safe_gate_bounds_match_oracle():
    raw_g = torch.randn(12, H, D) * 10.0
    a_log = torch.randn(H)
    mod = _module()
    g_mod = mod.safe_gate(raw_g, a_log, None)
    g_ref = kda_safe_gate(raw_g, a_log, None, lower_bound=KDA_GATE_LOWER_BOUND)
    torch.testing.assert_close(g_mod, g_ref, atol=1e-6, rtol=1e-6)
    # decay bound: exp(g) in (0, 1], g in [lower_bound, 0].
    assert (g_mod <= 0).all()
    assert (g_mod >= KDA_GATE_LOWER_BOUND).all()
    decay = torch.exp(g_mod)
    assert (decay > 0).all()
    assert (decay <= 1).all()
    # explicit floor: strongest possible forgetting is e^{-5}.
    assert decay.min().item() >= float(torch.exp(torch.tensor(KDA_GATE_LOWER_BOUND)))


# --------------------------------------------------------------------------- #
# end-to-end layer: conv -> gate -> recurrence -> gated RMSNorm
# --------------------------------------------------------------------------- #
def test_layer_parity_vs_oracle_fp32():
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

    mod = _module()
    out_mod = mod.forward(
        qkv, conv_weight=conv_w, raw_g=raw_g, beta_raw=beta_raw, a_log=a_log,
        dt_bias=dt_bias, g_out=g_out, o_norm_weight=o_w, output_dtype=torch.float32,
    )
    out_ref = kda_layer_reference(
        qkv, conv_weight=conv_w, raw_g=raw_g, beta_raw=beta_raw, a_log=a_log,
        dt_bias=dt_bias, g_out=g_out, o_norm_weight=o_w,
    )
    assert out_mod.shape == (T, H, D)
    torch.testing.assert_close(out_mod, out_ref, atol=FP32_ATOL, rtol=FP32_RTOL)


def test_layer_fp16_boundary_cast_tolerance():
    """Default output honors the dtype policy 'kda' cast site (float16)."""
    T = 12
    gen = torch.Generator().manual_seed(9)
    proj = H * D
    qkv = torch.randn(T, 3 * proj, generator=gen)
    conv_w = torch.randn(3 * proj, 4, generator=gen)
    raw_g = torch.randn(T, proj, generator=gen)
    beta_raw = torch.randn(T, H, generator=gen)
    a_log = torch.randn(H, generator=gen)
    dt_bias = torch.randn(proj, generator=gen)
    g_out = torch.randn(T, proj, generator=gen)
    o_w = torch.randn(D, generator=gen)

    mod = _module()
    out_fp16 = mod.forward(
        qkv, conv_weight=conv_w, raw_g=raw_g, beta_raw=beta_raw, a_log=a_log,
        dt_bias=dt_bias, g_out=g_out, o_norm_weight=o_w,
    )
    assert out_fp16.dtype == torch.float16
    out_ref = kda_layer_reference(
        qkv, conv_weight=conv_w, raw_g=raw_g, beta_raw=beta_raw, a_log=a_log,
        dt_bias=dt_bias, g_out=g_out, o_norm_weight=o_w,
    )
    torch.testing.assert_close(out_fp16.float(), out_ref, atol=FP16_ATOL, rtol=FP16_RTOL)


# --------------------------------------------------------------------------- #
# lazy package export (PEP 562) + Triton import hygiene
# --------------------------------------------------------------------------- #
def test_lazy_package_export():
    import vllm_ascend.models.glm5next_w2 as pkg

    assert "Glm5NextW2KDA" in dir(pkg)
    assert pkg.Glm5NextW2KDA is Glm5NextW2KDA


def test_kda_module_source_has_no_triton_import():
    """Grep-gate: no Triton import reachable in kda.py (comments are docs, not code).

    Mirrors ``test_glm5next_w2_package.py::test_package_source_has_no_triton_import``:
    case-sensitive lowercase ``triton``, strip trailing ``#`` comments, and treat
    any line that both imports and names triton as a hit.
    """
    hits = []
    for lineno, line in enumerate(_KDA_SRC.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]  # comments/docstrings are documentation
        imports_triton = (
            "import triton" in code
            or "from triton" in code
            or ("triton" in code and "import" in code)
            or "ops.triton" in code
        )
        if imports_triton:
            hits.append((lineno, line))
    assert not hits, f"triton import reachable in kda.py: {hits}"


def test_import_path_does_not_pull_shipped_triton_kda():
    """Importing the KDA module (lazily via the package) must not pull the shipped
    Triton KDA op / shipped glm5next base.

    NB: ``import vllm_ascend`` loads Triton globally via the plugin top-level
    init, so a bare ``'triton' in sys.modules`` check is not a valid signal here
    (see the package test's rationale). The honest gate is that *this* module's
    import path pulls neither the shipped base module nor its Triton KDA op.
    """
    code = (
        "import sys\n"
        "import vllm_ascend.models.glm5next_w2 as p\n"
        "m = p.Glm5NextW2KDA\n"  # force the lazy import of .kda
        "assert 'vllm_ascend.ops.triton.kda.kda' not in sys.modules, "
        "'shipped Triton KDA op pulled into glm5next_w2.kda import path'\n"
        "assert 'vllm_ascend.models.glm5next.kda' not in sys.modules, "
        "'shipped glm5next.kda pulled into import path'\n"
        "assert 'vllm_ascend.models.glm5next.model' not in sys.modules, "
        "'shipped glm5next base eagerly imported (should be lazy)'\n"
        "print('OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
    assert res.returncode == 0, f"import-hygiene subprocess failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"
    assert "OK" in res.stdout
