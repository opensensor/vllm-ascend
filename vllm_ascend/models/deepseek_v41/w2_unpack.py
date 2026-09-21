# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime W2->INT8 active-expert unpack + grouped QDQ MoE host math (E1.2).

This is the CPU *math* of the DeepSeek V4.1 2-bit (W2) routed-expert MoE that
E1.3 wires onto the Ascend 310P NPU INT8 grouped matmul. It has two pieces:

1. **Active-expert unpack** (:func:`unpack_active_experts`): given the top-``k``
   routing for a batch, only the *selected* experts are unpacked from the packed
   W2 bank into a **bounded cache** (one entry per unique active expert, never
   the full 384-expert bank). Each packed W2 weight
   (``uint8[out, in//4]`` codes + ``[out//32, in//32]`` per-block scale) is
   widened to signed int8 codes ``{-2,-1,0,1}`` and its per-``[32, 32]`` block
   scale is broadcast to the full ``[out, in]`` grid. The dequantized weight the
   grouped matmul consumes is ``codes.double() * block_scale`` -- the block scale
   is applied *into* the weight (before the matmul), the exact operand the E0.4
   reference ``w2_qdq_linear`` uses, so the two are bit-identical.

   The cache stores the ``(int8 codes, broadcast block scale)`` pair rather than
   the pre-multiplied float weight: that is the layout E1.3 hands the device
   INT8 grouped matmul (int8 weight code x per-block scale), and it keeps the
   application order explicit (scale multiplied element-wise with the codes,
   *not* folded into a single per-output-channel factor -- the block scale
   varies along the input axis every 32 columns and cannot be reduced to one).

2. **Grouped QDQ forward** (:func:`w2_active_moe_forward`): mirrors the
   T3.3-validated Qwen4Exp grouped path
   (``vllm_ascend/models/qwen4_exp/moe.py``): router ``softmax`` -> top-``k`` ->
   renormalize, per-token symmetric INT8 activation quant, one grouped GEMM per
   active expert (gate/up fused as ``w13`` -> SwiGLU -> ``w2`` down), router-weight
   scatter-add, and an always-on shared expert kept in higher precision.

Grouping mirrors the device ``group_list`` grouped-matmul semantics: (token,
slot) pairs are sorted by expert id via ``argsort``, each expert runs one batched
QDQ MLP over its whole group, and results scatter back with ``index_add_``. No
per-element ``tensor.item()`` in the hot path (AGENTS.md): a single boundary
sync (``.tolist()`` over the bounded unique-expert / group-count vectors), then
pure-tensor slices -- the same pattern as the Qwen4Exp reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tests.ut.deepseek_w2.reference.w2_moe_reference import W2Expert
from tests.ut.qwen38_1m.reference.w8a8_reference import (
    dequantize_per_token,
    quantize_per_token_int8,
)
from tools.deepseek_w2.w2_format import (
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    broadcast_block_scales,
    unpack_codes,
    unpack_w2_codes,
)

__all__ = [
    "ActiveExpertWeights",
    "unpack_active_experts",
    "route_topk_w2",
    "swiglu_gate_up",
    "w2_group_qdq_linear",
    "w2_active_moe_forward",
]


@dataclass
class ActiveExpertWeights:
    """Unpacked int8-code + block-scale operands for one active expert.

    The fused ``w13`` stacks gate rows ``[0, inter)`` above up rows
    ``[inter, 2*inter)`` (the SwiGLU fuse order); ``w2`` is the down projection.
    Each ``*_codes`` is signed int8 in ``{-2,-1,0,1}`` and each ``*_scale`` is the
    per-``[32, 32]`` block scale broadcast to the full weight grid. The
    dequantized weight is ``codes.double() * scale`` -- scale applied *before* the
    matmul (see :func:`w2_group_qdq_linear`).
    """

    w13_codes: torch.Tensor  # [2 * inter, hidden] int8
    w13_scale: torch.Tensor  # [2 * inter, hidden] float64
    w2_codes: torch.Tensor  # [hidden, inter] int8
    w2_scale: torch.Tensor  # [hidden, inter] float64

    def w13_weight(self) -> torch.Tensor:
        """Dequantized fused gate/up weight (codes x block scale), float64."""
        return self.w13_codes.double() * self.w13_scale

    def w2_weight(self) -> torch.Tensor:
        """Dequantized down weight (codes x block scale), float64."""
        return self.w2_codes.double() * self.w2_scale


def _unpack_w2_operand(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen packed W2 -> signed int8 codes + broadcast per-block scale.

    Returns ``(codes[int8, [out, in]], scale[float64, [out, in]])`` whose product
    is the dequantized weight. Uses the canonical E1.1 pack layout
    (``tools/deepseek_w2/w2_format.py``) so it is bit-identical to the E0.4
    reference unpack.
    """
    # Infer code width from the packed layout (W2 = 4 codes/byte, W4 = 2), so
    # mixed-precision W2/W4 expert banks unpack correctly with no extra plumbing.
    bits = 8 // (in_features // int(packed.shape[-1]))
    codes = unpack_codes(packed, in_features, bits)
    scale = broadcast_block_scales(block_scale, out_features, in_features, W2_BLOCK_ROWS, W2_BLOCK_COLS)
    return codes, scale


def unpack_active_experts(
    experts: list[W2Expert],
    topk_ids: torch.Tensor,
) -> dict[int, ActiveExpertWeights]:
    """Unpack **only** the routed (active) experts into a bounded cache.

    Args:
        experts: the full packed W2 expert bank (indexed by expert id).
        topk_ids: ``[T, top_k]`` selected expert ids for the batch.

    Returns:
        ``{expert_id: ActiveExpertWeights}`` holding exactly the unique experts
        that appear in ``topk_ids`` -- never the full bank. The gate/up weights
        are fused into ``w13``; the down weight is ``w2``.
    """
    # One boundary sync over the *bounded* active set (<= T * top_k, and at most
    # the number of distinct experts), never the 384-expert bank. torch.unique
    # returns the sorted distinct ids.
    active_ids = torch.unique(topk_ids).tolist()

    cache: dict[int, ActiveExpertWeights] = {}
    for expert_id in active_ids:
        expert = experts[expert_id]
        hidden = expert.hidden
        inter = expert.inter

        gate_codes, gate_scale = _unpack_w2_operand(expert.gate_packed, expert.gate_scale, inter, hidden)
        up_codes, up_scale = _unpack_w2_operand(expert.up_packed, expert.up_scale, inter, hidden)
        down_codes, down_scale = _unpack_w2_operand(expert.down_packed, expert.down_scale, hidden, inter)

        cache[expert_id] = ActiveExpertWeights(
            w13_codes=torch.cat([gate_codes, up_codes], dim=0),
            w13_scale=torch.cat([gate_scale, up_scale], dim=0),
            w2_codes=down_codes,
            w2_scale=down_scale,
        )
    return cache


def route_topk_w2(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router: ``softmax`` -> top-``k`` -> optional renormalize.

    Matches the E0.4 reference ``route_topk`` bit-for-bit when
    ``renormalize=True`` (float64 softmax, then divide the kept probabilities by
    their per-token sum). ``renormalize=False`` leaves the raw top-``k``
    probabilities (they sum to < 1) -- used only to prove the renorm has teeth.

    Returns ``(expert_ids[int64, [T, top_k]], weights[float64, [T, top_k]])``.
    """
    probs = torch.softmax(router_logits.double(), dim=-1)
    top_w, top_ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    return top_ids, top_w


def swiglu_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """Split the fused ``w13`` output in half -> ``silu(gate) * up``.

    Columns ``[0, inter)`` are gate, ``[inter, 2*inter)`` are up (the fuse order
    in :func:`unpack_active_experts`).
    """
    inter = gate_up.shape[-1] // 2
    gate = gate_up[..., :inter]
    up = gate_up[..., inter:]
    return F.silu(gate) * up


def w2_group_qdq_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """One grouped W2 linear: ``dequant(quant(x)) @ (codes * scale).T``.

    The per-block ``scale`` is applied element-wise into the int8 ``codes``
    *before* the matmul -- the block scale varies along the input axis every 32
    columns, so it cannot be pulled out as a single per-output-channel factor.
    The activation path is the shared symmetric per-token INT8 QDQ, identical to
    the E0.4 W2 reference and the Qwen4Exp W8A8 grouped path.
    """
    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a).double()
    w_deq = codes.double() * scale
    return x_deq @ w_deq.t()


def w2_active_moe_forward(
    x: torch.Tensor,
    experts: list[W2Expert],
    router_logits: torch.Tensor,
    top_k: int,
    shared_expert: W2Expert | None = None,
    *,
    renormalize: bool = True,
    cache: dict[int, ActiveExpertWeights] | None = None,
) -> torch.Tensor:
    """Grouped W2->INT8 active-expert QDQ MoE forward.

    Routes the batch, unpacks only the active experts (bounded cache), runs one
    grouped QDQ MLP per expert (fused ``w13`` -> SwiGLU -> ``w2``), scales by the
    router weight, scatters back, then adds the higher-precision shared expert.

    Args:
        x: ``[T, hidden]`` float activations.
        experts: full packed W2 expert bank (indexed by expert id).
        router_logits: ``[T, num_experts]``.
        top_k: experts kept per token.
        shared_expert: optional always-on expert added to every token.
        renormalize: renormalize the top-``k`` router weights (default True).
        cache: optional pre-built active-expert cache (else built internally).

    Returns:
        ``[T, hidden]`` float64 output.
    """
    x = x.double()
    num_tokens, hidden = x.shape
    expert_ids, weights = route_topk_w2(router_logits, top_k, renormalize=renormalize)

    if cache is None:
        cache = unpack_active_experts(experts, expert_ids)

    pair_expert = expert_ids.reshape(-1)  # [T * top_k]
    pair_weight = weights.reshape(-1, 1)
    pair_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    pair_x = x[pair_token]

    order = torch.argsort(pair_expert, stable=True)
    sorted_expert = pair_expert[order]
    sorted_x = pair_x[order]
    sorted_weight = pair_weight[order]
    sorted_token = pair_token[order]

    # Grouped boundaries: one boundary sync over the bounded unique-expert /
    # group-count vectors, then pure-tensor slices (no per-element item()).
    uniq_expert, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
    out = torch.zeros(num_tokens, hidden, dtype=torch.float64, device=x.device)

    start = 0
    for expert_id, count in zip(uniq_expert.tolist(), counts.tolist()):
        stop = start + count
        weight = cache[expert_id]
        group_x = sorted_x[start:stop]
        gate_up = w2_group_qdq_linear(group_x, weight.w13_codes, weight.w13_scale)
        hidden_act = swiglu_gate_up(gate_up)
        y = w2_group_qdq_linear(hidden_act, weight.w2_codes, weight.w2_scale)
        y = y * sorted_weight[start:stop]
        out.index_add_(0, sorted_token[start:stop], y)
        start = stop

    if shared_expert is not None:
        out = out + shared_expert.forward(x)
    return out
