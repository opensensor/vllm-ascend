# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side W8A8 fused-MoE forward for the Ascend 310P Qwen4Exp path (T3.x).

This is the CPU *math* of the routed-expert W8A8_DYNAMIC fused MoE that runs on
device through :class:`AscendW8A8DynamicFusedMoEMethod310`
(``vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py``). That method calls
``torch_npu.npu_quant_grouped_matmul_dequant`` + ``npu_swiglu``, which are not
importable off-NPU (``torch_npu`` is absent on the host); this module re-expresses
the identical quantize/dequantize (QDQ) grouped-matmul math in pure PyTorch so the
310P assembly can run and be validated host-side.

The math is the frozen, T3.3-validated formulation
(``tests/ut/qwen38_1m/test_moe_w8a8_parity.py``):

  * router: ``softmax`` -> ``top_k`` -> optional renormalize (``w / w.sum(-1)``)
    -> ``* routed_scaling_factor`` (mirrors ``AscendGroupedTopKRouter310`` /
    the fork ``Qwen3NextSparseMoeBlock`` ``norm_topk_prob`` semantics);
  * per-token symmetric INT8 activation quant (``quant_mode="pertoken"``:
    ``scale = amax(|row|)/127``, no activation offset);
  * per-output-channel INT8 weight dequant ``(q - offset) * scale`` -- the
    checkpoint order (offset subtracted first). The real 300i experts are
    *symmetric* (offset == 0), so the subtraction is a no-op on real data but the
    kernel order is preserved exactly;
  * fused gate/up (``w13``) GEMM -> ``swiglu`` -> down (``w2``) GEMM per expert.

Grouping mirrors the device ``group_list`` grouped-matmul semantics: (token, slot)
pairs are sorted by expert id and each expert runs *one* batched QDQ GEMM over its
whole group, then results are scaled by the router weight and scattered back with
``index_add_``. No per-element ``tensor.item()`` in the hot path (AGENTS.md): the
grouping uses ``argsort`` / ``bincount`` and a single boundary ``.tolist()``.

The shared expert stays non-quantized F16 (per the T3.1 weight-mapping contract:
router / shared-expert / attention / lm_head / embeddings / PLE are not W8A8) and
is applied densely and unweighted to every token, added to the routed output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "dequantize_weight_perchannel",
    "quantize_activation_per_token",
    "route_topk",
    "w8a8_grouped_experts",
    "w8a8_qdq_linear",
    "swiglu_gate_up",
]

# Symmetric per-token INT8 grid uses the positive half-range (127 levels), no
# activation offset (the 310P W8A8 activation quant is symmetric).
_INT8_SYM_LEVELS = 127.0
_INT8_MIN = -128
_INT8_MAX = 127


def quantize_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric per-token INT8 activation quant (``quant_mode="pertoken"``).

    Returns ``(q_int8, scale)`` with ``scale`` shaped ``[T, 1]``. All-zero rows get
    ``scale == 1`` to avoid a divide-by-zero (they quantize to all-zero anyway).
    """
    x = x.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _INT8_SYM_LEVELS, torch.ones_like(amax))
    q = torch.round(x / scale).clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
    return q, scale


def dequantize_weight_perchannel(
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_offset: torch.Tensor,
) -> torch.Tensor:
    """Per-output-channel INT8 weight dequant ``(q - offset) * scale``.

    ``weight_scale`` / ``weight_offset`` are ``[out, 1]``; offset is subtracted
    first (the checkpoint order). Symmetric checkpoints carry ``offset == 0``.
    """
    return (weight_int8.to(torch.float32) - weight_offset.to(torch.float32)) * weight_scale.to(torch.float32)


def w8a8_qdq_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_offset: torch.Tensor,
) -> torch.Tensor:
    """One W8A8 dynamic linear: ``dequant(quant(x)) @ dequant(w).T``.

    Mirrors ``npu_quant_grouped_matmul_dequant(quant_mode="pertoken")`` for a
    single (already-grouped) activation batch and one expert's weight.
    """
    q_a, scale_a = quantize_activation_per_token(x)
    x_deq = q_a.to(torch.float32) * scale_a
    w_deq = dequantize_weight_perchannel(weight_int8, weight_scale, weight_offset)
    return x_deq @ w_deq.t()


def swiglu_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """``npu_swiglu``: split the last dim in half -> ``silu(gate) * up``.

    ``w13`` rows ``[0, moe)`` are gate, ``[moe, 2*moe)`` are up (T3.1 fuse order),
    so the gmm1 output columns split the same way.
    """
    moe = gate_up.shape[-1] // 2
    gate = gate_up[..., :moe]
    up = gate_up[..., moe:]
    return F.silu(gate) * up


def route_topk(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router: ``softmax`` -> ``top_k`` -> optional renorm -> ``* scaling_factor``.

    Returns ``(topk_weights[float32], topk_ids[int64])``. Renormalization divides
    the selected probabilities by their per-token sum so they sum to
    ``routed_scaling_factor`` (matching ``norm_topk_prob`` in the fork MoE block).
    """
    probs = router_logits.to(torch.float32).softmax(dim=-1)
    topk_weights, topk_ids = probs.topk(top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int64)


def w8a8_grouped_experts(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_offset: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_offset: torch.Tensor,
) -> torch.Tensor:
    """Grouped routed-expert W8A8 forward (``group_list`` grouped-matmul mimic).

    Every (token, slot) pair is sorted by its selected expert id; each expert
    evaluates its whole group in one QDQ MLP call (gmm1 -> swiglu -> gmm2), the
    result is scaled by the router weight and scattered back with ``index_add_``.
    Empty groups (skew) are skipped. All arithmetic is float32; the caller casts.

    Args:
        x: ``[T, hidden]`` activations (float; upcast to float32 internally).
        topk_weights: ``[T, top_k]`` router weights (already renormalized/scaled).
        topk_ids: ``[T, top_k]`` selected expert ids.
        w13_weight: ``[E, 2*moe, hidden]`` int8 fused gate/up weights.
        w13_weight_scale / w13_weight_offset: ``[E, 2*moe, 1]`` per-channel params.
        w2_weight: ``[E, hidden, moe]`` int8 down weights.
        w2_weight_scale / w2_weight_offset: ``[E, hidden, 1]`` per-channel params.

    Returns:
        ``[T, hidden]`` float32 routed-expert output.
    """
    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    num_experts = w13_weight.shape[0]
    x32 = x.to(torch.float32)

    pair_expert = topk_ids.reshape(-1)  # [T * top_k]
    pair_weight = topk_weights.to(torch.float32).reshape(-1, 1)
    pair_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    pair_x = x32[pair_token]  # gather activations for each (token, slot) pair

    order = torch.argsort(pair_expert, stable=True)
    sorted_expert = pair_expert[order]
    sorted_x = pair_x[order]
    sorted_weight = pair_weight[order]
    sorted_token = pair_token[order]

    counts = torch.bincount(sorted_expert, minlength=num_experts)
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)

    start = 0
    # Single boundary sync (AGENTS.md): one .tolist() over the group counts, then
    # a pure-tensor slice per non-empty expert group -- no per-element item().
    for expert_id, count in enumerate(counts.tolist()):
        if count == 0:  # expert received no tokens (skew): empty group, skip.
            continue
        stop = start + count
        group_x = sorted_x[start:stop]
        gate_up = w8a8_qdq_linear(
            group_x,
            w13_weight[expert_id],
            w13_weight_scale[expert_id],
            w13_weight_offset[expert_id],
        )
        hidden_act = swiglu_gate_up(gate_up)
        y = w8a8_qdq_linear(
            hidden_act,
            w2_weight[expert_id],
            w2_weight_scale[expert_id],
            w2_weight_offset[expert_id],
        )
        y = y * sorted_weight[start:stop]
        out.index_add_(0, sorted_token[start:stop], y)
        start = stop

    return out
