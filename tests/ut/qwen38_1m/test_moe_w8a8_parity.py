# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 fused-MoE parity vs the eager T0.6 QDQ reference (T3.3).

Host-side (CPU) parity harness for the Ascend 310P Qwen4Exp W8A8_DYNAMIC fused
MoE. There is no NPU / ``torch_npu`` here, so the device kernels
(``npu_quant_grouped_matmul_dequant`` + ``npu_swiglu`` inside
``vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py``
``AscendW8A8DynamicFusedMoEMethod310``) are re-expressed as their *math* on
CPU using the frozen T0.6 QDQ primitives
(``tests/ut/qwen38_1m/reference/w8a8_reference.py``):

  * per-token symmetric INT8 activation quant (``quant_mode="pertoken"``),
  * per-output-channel INT8 weight dequant ``(q - offset) * scale`` (offset
    subtracted first -- the checkpoint order), and
  * swiglu on the fused gate/up (``w13``) projection, then the down (``w2``)
    projection.

Two independent formulations of the same MoE forward are compared:

  * ``_moe_eager`` -- the reference: a plain per-token loop, each selected
    expert evaluated on a single-row activation (the T0.6 "eager" altitude).
  * ``_moe_fused`` -- the device-path mimic: (token, slot) pairs are grouped
    by expert id and each expert runs *one batched* QDQ GEMM over its whole
    group (the ``group_list`` grouped-matmul semantics), then results are
    scaled by the router weight and scattered back. No ``tensor.item()`` in the
    grouped hot path (AGENTS.md) -- grouping uses ``argsort`` / ``bincount`` and
    a single ``.tolist()`` boundary sync.

The two must agree at the pre-declared W8A8 GEMM tolerances. On top of the base
parity we cover, per the T3.3 acceptance:

  * expert-distribution skew (all tokens routed to a single expert),
  * router renormalization (``w / w.sum(-1)`` then ``* routed_scaling_factor``;
    a guard proves skipping it materially changes the output), and
  * offset application order (a guard proves ``(q + offset) * scale`` -- offset
    *added* -- breaks parity whenever the offset is non-zero).

The real 300i checkpoint uses *symmetric* expert weights (offset == 0), so the
order is a no-op on real data; an optional, CI-safe realism check asserts that
against 3 real experts when the model mount is present, and is skipped
otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.ut.qwen38_1m.reference.tolerances import (
    W8A8_GEMM_ATOL,
    W8A8_GEMM_RTOL,
)
from tests.ut.qwen38_1m.reference.w8a8_reference import (
    INT8_MAX,
    dequantize_per_token,
    dequantize_weight,
    quantize_per_token_int8,
)

# --------------------------------------------------------------------------- #
# Pre-declared tolerances (PRD Sec 8.1: named constants before any assertion).
# --------------------------------------------------------------------------- #
# The eager loop and the grouped/batched path run the *same* QDQ math on the
# same per-token scales and the same dequantized weights; only FP32 matmul
# reassociation and the summation order over a token's selected experts differ.
# Reuse the T0.6 W8A8 GEMM bounds -- they were chosen for exactly this
# "same GEMM computed two ways" comparison.
MOE_PARITY_RTOL = W8A8_GEMM_RTOL
MOE_PARITY_ATOL = W8A8_GEMM_ATOL

# A wrong offset order / a dropped renormalization is not a rounding effect: it
# shifts the output by O(1). A guard must see divergence far above the parity
# floor to prove the test has teeth.
GUARD_MIN_DIVERGENCE = 1e-2

# Renormalized top-k weights sum to the routed scaling factor per token; this is
# exact FP32 arithmetic up to rounding.
RENORM_SUM_ATOL = 1e-5

# --------------------------------------------------------------------------- #
# Frozen Qwen4Exp W8A8 MoE geometry (real 300i checkpoint).
#   hidden=2560, moe_intermediate=640, 512 experts, top-10, 1 shared expert.
# The parity math is dimension-independent, so the CI harness uses small
# hidden/moe dims but keeps the *logical* expert/top-k/shared structure so the
# grouping, skew and scatter paths are exercised exactly as on the real model.
# --------------------------------------------------------------------------- #
REAL_NUM_EXPERTS = 512
REAL_TOP_K = 10
REAL_NUM_SHARED_EXPERTS = 1

REAL_CHECKPOINT = Path(
    "/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/models/ascend/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i"
)


# --------------------------------------------------------------------------- #
# QDQ primitives (device-kernel math, offset order selectable for the guard).
# --------------------------------------------------------------------------- #
def _qdq_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: torch.Tensor,
    *,
    add_offset: bool = False,
) -> torch.Tensor:
    """One W8A8 dynamic linear: per-token quant(x) @ dequant(w).T.

    Mirrors ``npu_quant_grouped_matmul_dequant(quant_mode="pertoken")``: the
    activation is dynamically per-token symmetric INT8 quantized and the INT8
    weight is dequantized per output channel. ``add_offset`` selects the *wrong*
    order ``(q + offset) * scale`` for the offset-order guard; the default is the
    checkpoint order ``(q - offset) * scale`` (T0.6 ``dequantize_weight``).
    """
    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a)
    if add_offset:
        w_deq = (w_q.to(torch.float32) + w_offset) * w_scale
    else:
        w_deq = dequantize_weight(w_q, w_scale, w_offset)
    return x_deq @ w_deq.t()


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """npu_swiglu: split last dim in half -> silu(gate) * up.

    ``w13`` rows [0, moe) are the gate projection, rows [moe, 2*moe) the up
    projection (weight_mapping T3.1), so column [0, moe) is gate and [moe, 2*moe)
    is up in the fused gmm1 output.
    """
    moe = gate_up.shape[-1] // 2
    gate = gate_up[..., :moe]
    up = gate_up[..., moe:]
    return torch.nn.functional.silu(gate) * up


def _expert_mlp(
    x: torch.Tensor,
    expert: _ExpertWeights,
    *,
    add_offset: bool = False,
) -> torch.Tensor:
    """gmm1 (gate/up) -> swiglu -> gmm2 (down) for one expert on a token batch."""
    gate_up = _qdq_linear(x, expert.w13_q, expert.w13_scale, expert.w13_offset, add_offset=add_offset)
    hidden = _swiglu(gate_up)
    return _qdq_linear(hidden, expert.w2_q, expert.w2_scale, expert.w2_offset, add_offset=add_offset)


# --------------------------------------------------------------------------- #
# Weight containers + builders (symmetric = real scheme, asymmetric = guard).
# --------------------------------------------------------------------------- #
class _ExpertWeights:
    __slots__ = (
        "w13_q",
        "w13_scale",
        "w13_offset",
        "w2_q",
        "w2_scale",
        "w2_offset",
    )

    def __init__(self, w13_q, w13_scale, w13_offset, w2_q, w2_scale, w2_offset):
        self.w13_q = w13_q
        self.w13_scale = w13_scale
        self.w13_offset = w13_offset
        self.w2_q = w2_q
        self.w2_scale = w2_scale
        self.w2_offset = w2_offset


def _quantize_symmetric(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8 (offset == 0): the real 300i scheme."""
    w = w.float()
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / INT8_MAX, torch.ones_like(amax))
    q = torch.round(w / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    offset = torch.zeros_like(scale)
    return q, scale, offset


def _quantize_asymmetric(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel asymmetric INT8 (offset != 0) for the order guard.

    Uses the T0.6 asymmetric qparams so the dequant ``(q - offset) * scale``
    is self-consistent with the quantized weight.
    """
    from tests.ut.qwen38_1m.reference.w8a8_reference import (
        compute_weight_qparams,
        quantize_weight_int8,
    )

    scale, offset = compute_weight_qparams(w)
    q = quantize_weight_int8(w, scale, offset)
    return q, scale, offset


def _build_experts(
    num_experts: int,
    hidden: int,
    moe: int,
    seed: int,
    *,
    asymmetric: bool = False,
) -> list[_ExpertWeights]:
    gen = torch.Generator().manual_seed(seed)
    quant = _quantize_asymmetric if asymmetric else _quantize_symmetric
    experts: list[_ExpertWeights] = []
    for _ in range(num_experts):
        # w13 fuses gate+up -> [2*moe, hidden]; w2 is down -> [hidden, moe].
        w13 = torch.randn(2 * moe, hidden, generator=gen) * 0.2
        w2 = torch.randn(hidden, moe, generator=gen) * 0.2
        w13_q, w13_scale, w13_offset = quant(w13)
        w2_q, w2_scale, w2_offset = quant(w2)
        experts.append(_ExpertWeights(w13_q, w13_scale, w13_offset, w2_q, w2_scale, w2_offset))
    return experts


# --------------------------------------------------------------------------- #
# Router (mirrors AscendGroupedTopKRouter310: softmax -> topk -> renorm -> scale)
# --------------------------------------------------------------------------- #
def _route(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = router_logits.to(torch.float32).softmax(dim=-1)
    topk_weights, topk_ids = probs.topk(top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int64)


# --------------------------------------------------------------------------- #
# MoE forward: eager reference vs grouped/fused device-path mimic.
# --------------------------------------------------------------------------- #
def _moe_eager(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: list[_ExpertWeights],
    shared: list[_ExpertWeights],
    *,
    add_offset: bool = False,
) -> torch.Tensor:
    """Per-token reference loop (the eager QDQ altitude).

    A single batched ``.tolist()`` moves routing to Python (the AGENTS.md
    "batch operations -> single sync" pattern); the arithmetic per token is
    the plain QDQ MLP.
    """
    num_tokens, hidden = x.shape
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32)
    ids = topk_ids.tolist()
    weights = topk_weights.to(torch.float32).tolist()
    for t in range(num_tokens):
        row = x[t : t + 1]
        acc = torch.zeros(1, hidden, dtype=torch.float32)
        for slot, expert_id in enumerate(ids[t]):
            y = _expert_mlp(row, experts[expert_id], add_offset=add_offset)
            acc = acc + weights[t][slot] * y
        out[t] = acc[0]
    # Shared expert(s): dense, applied to every token, unweighted.
    for shared_expert in shared:
        out = out + _expert_mlp(x, shared_expert, add_offset=add_offset)
    return out


def _moe_fused(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: list[_ExpertWeights],
    shared: list[_ExpertWeights],
    *,
    add_offset: bool = False,
) -> torch.Tensor:
    """Grouped/batched device-path mimic (group_list grouped-matmul semantics).

    (token, slot) pairs are sorted by expert id; each expert evaluates its whole
    group in one QDQ MLP call, results are scaled by the router weight and
    scattered back with ``index_add_``. Grouping is pure-tensor (argsort /
    bincount) with a single boundary ``.tolist()`` -- no per-element
    ``tensor.item()`` in the hot path.
    """
    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]

    pair_expert = topk_ids.reshape(-1)  # [T * top_k]
    pair_weight = topk_weights.to(torch.float32).reshape(-1, 1)
    pair_token = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    pair_x = x[pair_token]  # gather activations for each (token, slot) pair

    order = torch.argsort(pair_expert, stable=True)
    sorted_expert = pair_expert[order]
    sorted_x = pair_x[order]
    sorted_weight = pair_weight[order]
    sorted_token = pair_token[order]

    counts = torch.bincount(sorted_expert, minlength=len(experts))
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32)

    start = 0
    for expert_id, count in enumerate(counts.tolist()):
        if count == 0:  # expert received no tokens (skew): empty group, skip.
            continue
        stop = start + count
        group_x = sorted_x[start:stop]
        y = _expert_mlp(group_x, experts[expert_id], add_offset=add_offset)
        y = y * sorted_weight[start:stop]
        out.index_add_(0, sorted_token[start:stop], y)
        start = stop

    for shared_expert in shared:
        out = out + _expert_mlp(x, shared_expert, add_offset=add_offset)
    return out


# --------------------------------------------------------------------------- #
# Fixtures / helpers.
# --------------------------------------------------------------------------- #
def _rand_inputs(num_tokens, hidden, num_experts, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=gen)
    router_logits = torch.randn(num_tokens, num_experts, generator=gen)
    return x, router_logits


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_experts, top_k, num_shared",
    [
        (REAL_NUM_EXPERTS, REAL_TOP_K, REAL_NUM_SHARED_EXPERTS),  # real geometry
        (8, 3, 1),
        (16, 4, 0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_matches_eager_qdq(num_experts, top_k, num_shared, seed):
    """Grouped/batched fused MoE == eager per-token QDQ reference."""
    hidden, moe, num_tokens = 16, 8, 24
    x, router_logits = _rand_inputs(num_tokens, hidden, num_experts, seed)
    topk_weights, topk_ids = _route(router_logits, top_k)

    experts = _build_experts(num_experts, hidden, moe, seed=seed + 100)
    shared = _build_experts(num_shared, hidden, moe, seed=seed + 900)

    y_eager = _moe_eager(x, topk_weights, topk_ids, experts, shared)
    y_fused = _moe_fused(x, topk_weights, topk_ids, experts, shared)

    torch.testing.assert_close(y_fused, y_eager, rtol=MOE_PARITY_RTOL, atol=MOE_PARITY_ATOL)


def test_skew_all_tokens_to_one_expert():
    """Expert-distribution skew: every token routes to the same expert.

    All but one grouped bucket is empty; the fused grouping must still match the
    eager reference.
    """
    num_experts, top_k, hidden, moe, num_tokens = 512, 10, 16, 8, 20
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(num_tokens, hidden, generator=gen)

    # Force the router onto a single top-1 expert (id 42) for every token by
    # making its logit dominate; the remaining top-k slots fill deterministically.
    router_logits = torch.randn(num_tokens, num_experts, generator=gen)
    router_logits[:, 42] = 50.0
    topk_weights, topk_ids = _route(router_logits, top_k)
    assert torch.all(topk_ids[:, 0] == 42)

    experts = _build_experts(num_experts, hidden, moe, seed=11)
    shared = _build_experts(1, hidden, moe, seed=12)

    y_eager = _moe_eager(x, topk_weights, topk_ids, experts, shared)
    y_fused = _moe_fused(x, topk_weights, topk_ids, experts, shared)
    torch.testing.assert_close(y_fused, y_eager, rtol=MOE_PARITY_RTOL, atol=MOE_PARITY_ATOL)

    # Degenerate extreme: literally all top-k slots on one expert -> one non-empty
    # group of size num_tokens*top_k, all others empty.
    single_ids = torch.full((num_tokens, top_k), 5, dtype=torch.int64)
    single_weights = torch.full((num_tokens, top_k), 1.0 / top_k)
    y_eager1 = _moe_eager(x, single_weights, single_ids, experts, shared)
    y_fused1 = _moe_fused(x, single_weights, single_ids, experts, shared)
    torch.testing.assert_close(y_fused1, y_eager1, rtol=MOE_PARITY_RTOL, atol=MOE_PARITY_ATOL)


def test_router_renormalization():
    """Renormalized weights sum to the scaling factor, and renorm changes output.

    Guard: reusing *un-renormalized* weights against the renormalized reference
    must diverge far beyond the parity floor, proving renorm is really applied.
    """
    num_experts, top_k, hidden, moe, num_tokens = 32, 6, 16, 8, 16
    x, router_logits = _rand_inputs(num_tokens, hidden, num_experts, seed=3)

    scaling = 1.5
    weights_renorm, ids = _route(router_logits, top_k, renormalize=True, routed_scaling_factor=scaling)
    weights_raw, ids_raw = _route(router_logits, top_k, renormalize=False, routed_scaling_factor=scaling)
    assert torch.equal(ids, ids_raw)

    # Renormalized top-k weights sum to the routed scaling factor per token.
    per_token_sum = weights_renorm.sum(dim=-1)
    torch.testing.assert_close(per_token_sum, torch.full_like(per_token_sum, scaling), atol=RENORM_SUM_ATOL, rtol=0)
    # Raw softmax top-k weights do NOT (their pre-scale sum is < 1).
    assert torch.all(weights_raw.sum(dim=-1) < scaling - GUARD_MIN_DIVERGENCE)

    experts = _build_experts(num_experts, hidden, moe, seed=44)
    shared = _build_experts(1, hidden, moe, seed=45)

    y_renorm = _moe_fused(x, weights_renorm, ids, experts, shared)
    y_eager = _moe_eager(x, weights_renorm, ids, experts, shared)
    torch.testing.assert_close(y_renorm, y_eager, rtol=MOE_PARITY_RTOL, atol=MOE_PARITY_ATOL)

    # Guard: dropping renormalization materially changes the MoE output.
    y_raw = _moe_fused(x, weights_raw, ids, experts, shared)
    assert _max_abs_err(y_raw, y_renorm) > GUARD_MIN_DIVERGENCE


def test_offset_application_order():
    """Offset order guard on asymmetric experts.

    With non-zero per-channel offsets, the checkpoint order ``(q - offset) *
    scale`` gives fused/eager parity, while the wrong order ``(q + offset) *
    scale`` diverges far beyond the parity floor.
    """
    num_experts, top_k, hidden, moe, num_tokens = 16, 4, 16, 8, 16
    x, router_logits = _rand_inputs(num_tokens, hidden, num_experts, seed=5)
    topk_weights, topk_ids = _route(router_logits, top_k)

    experts = _build_experts(num_experts, hidden, moe, seed=55, asymmetric=True)
    shared = _build_experts(1, hidden, moe, seed=56, asymmetric=True)

    # Sanity: the guard is only meaningful if offsets are actually non-zero.
    assert experts[0].w13_offset.abs().max().item() > 0.5

    # Correct order: fused == eager.
    y_fused = _moe_fused(x, topk_weights, topk_ids, experts, shared)
    y_eager = _moe_eager(x, topk_weights, topk_ids, experts, shared)
    torch.testing.assert_close(y_fused, y_eager, rtol=MOE_PARITY_RTOL, atol=MOE_PARITY_ATOL)

    # Wrong order (offset added): must break parity against the correct reference.
    y_wrong = _moe_fused(x, topk_weights, topk_ids, experts, shared, add_offset=True)
    assert _max_abs_err(y_wrong, y_eager) > GUARD_MIN_DIVERGENCE


# --------------------------------------------------------------------------- #
# Optional realism check against the real 300i checkpoint (skipped in CI).
# --------------------------------------------------------------------------- #
def _load_real_expert(layer: int, expert: int) -> _ExpertWeights | None:
    try:
        from safetensors import safe_open
    except Exception:
        return None
    import json

    index_path = REAL_CHECKPOINT / "quant_model_weights.safetensors.index.json"
    if not index_path.is_file():
        return None
    weight_map = json.loads(index_path.read_text())["weight_map"]

    def load(name: str) -> torch.Tensor:
        with safe_open(str(REAL_CHECKPOINT / weight_map[name]), framework="pt") as f:
            return f.get_tensor(name)

    base = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    gate_q = load(f"{base}.gate_proj.weight")
    up_q = load(f"{base}.up_proj.weight")
    gate_s = load(f"{base}.gate_proj.weight_scale")
    up_s = load(f"{base}.up_proj.weight_scale")
    gate_o = load(f"{base}.gate_proj.weight_offset")
    up_o = load(f"{base}.up_proj.weight_offset")
    w2_q = load(f"{base}.down_proj.weight")
    w2_s = load(f"{base}.down_proj.weight_scale")
    w2_o = load(f"{base}.down_proj.weight_offset")
    # Fuse gate/up column-wise into w13 (gate rows first, then up) per T3.1.
    w13_q = torch.cat([gate_q, up_q], dim=0)
    w13_scale = torch.cat([gate_s, up_s], dim=0)
    w13_offset = torch.cat([gate_o, up_o], dim=0)
    return _ExpertWeights(w13_q, w13_scale, w13_offset, w2_q, w2_s, w2_o)


@pytest.mark.skipif(
    not (REAL_CHECKPOINT / "quant_model_weights.safetensors.index.json").is_file()
    or os.environ.get("QWEN4EXP_SKIP_REAL_CKPT") == "1",
    reason="real 300i checkpoint mount not present (CI-safe skip)",
)
def test_real_checkpoint_experts_are_symmetric():
    """Realism: real experts are symmetric int8 [out,in]; QDQ MLP runs finite.

    Confirms the offset-order guard is a no-op on real data (offset == 0) and
    that the fused expert math executes on genuine checkpoint tensors.
    """
    experts = [_load_real_expert(0, e) for e in (0, 1, 2)]
    experts = [e for e in experts if e is not None]
    if not experts:
        pytest.skip("could not load real experts (safetensors unavailable)")

    hidden = experts[0].w13_q.shape[1]
    assert hidden == 2560  # frozen geometry
    assert experts[0].w13_q.shape[0] == 2 * 640  # 2 * moe_intermediate_size
    assert experts[0].w2_q.shape == (2560, 640)

    for expert in experts:
        assert expert.w13_q.dtype == torch.int8
        assert expert.w2_q.dtype == torch.int8
        # Symmetric scheme: offsets are exactly zero.
        assert torch.count_nonzero(expert.w13_offset) == 0
        assert torch.count_nonzero(expert.w2_offset) == 0
        assert torch.all(expert.w13_scale > 0)
        assert torch.all(expert.w2_scale > 0)
        # With offset == 0, the correct and "wrong" orders coincide (no-op).
        w_sub = dequantize_weight(expert.w2_q, expert.w2_scale, expert.w2_offset)
        w_add = (expert.w2_q.to(torch.float32) + expert.w2_offset) * expert.w2_scale
        torch.testing.assert_close(w_sub, w_add)

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, hidden, generator=gen)
    y = _expert_mlp(x, experts[0])
    assert y.shape == (4, hidden)
    assert torch.isfinite(y).all()
