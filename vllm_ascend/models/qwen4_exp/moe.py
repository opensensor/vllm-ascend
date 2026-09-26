# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 fused-MoE forward for the Ascend 310P Qwen4Exp path (T3.x).

The NPU path groups routes by expert and evaluates the packed bank with the
310P dynamic-quant grouped matmul. The weight-only per-expert path remains a
compatibility fallback for unpacked banks. This module retains the pure-PyTorch
QDQ math as the host reference.

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

Prefill grouping mirrors the device ``group_list`` grouped-matmul semantics:
(token, slot) pairs are sorted by expert id and each expert runs one batched
dynamic-quant GEMM over its group, then results are scaled by the router weight
and gathered back in the original route order. The packed NPU path keeps group
boundaries and top-k ids on the device, including single-token decode. The
unpacked compatibility path retains a single ``.tolist()`` boundary.

The shared expert stays non-quantized F16 (per the T3.1 weight-mapping contract:
router / shared-expert / attention / lm_head / embeddings / PLE are not W8A8) and
is applied densely and unweighted to every token, added to the routed output.

Expert-dimension TP slicing: when the bank only holds a contiguous slice of the
global experts, the grouped forward returns the rank's *partial* sum (peer-owned
top-k slots are dropped); the caller all-reduces the partials across ranks and
then adds the replicated shared expert exactly once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .grouped_expert_dispatch import build_grouped_expert_dispatch

__all__ = [
    "dequantize_weight_perchannel",
    "quantize_activation_per_token",
    "route_topk",
    "w8a8_grouped_experts",
    "w8a8_grouped_experts_npu",
    "w8a8_qdq_linear",
    "swiglu_gate_up",
]

# Symmetric per-token INT8 grid uses the positive half-range (127 levels), no
# activation offset (the 310P W8A8 activation quant is symmetric).
_INT8_SYM_LEVELS = 127.0
_INT8_MIN = -128
_INT8_MAX = 127
_PACKED_LOCAL_ROUTE_MIN_TOKENS = 128
_FUSED_ROUTING_MAX_TOKENS = 2


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


def _w8a16_linear_npu(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run a per-channel W8A16 linear without materializing FP16 weights.

    Qwen4Exp's shipped expert weights are symmetric (zero offset). 310P's
    weight-only matmul accepts FP16 activations and a contiguous post-load
    ``[K, N]`` INT8 weight, preserving the accurate A16 path while avoiding the per-call
    INT8->FP16 conversion and scale multiplication used by the compatibility
    fallback.
    """
    import torch_npu

    return torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight_int8,
        antiquant_scale=weight_scale.reshape(-1).to(x.dtype),
        antiquant_offset=None,
    )


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


def _w8a16_decode_experts_npu(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    expert_offset: int,
) -> torch.Tensor:
    """Evaluate a single token without sorting its few selected expert ids.

    The host needs the selected ids to dispatch weight-only expert matmuls.
    Copy just the top-k ids at that existing synchronization boundary, then
    keep all activations, route weights, and accumulated outputs on the NPU.
    """
    import torch_npu

    num_local_experts = len(w13_weight)
    route_outputs = torch.zeros(topk_ids.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
    for route_index, expert_id in enumerate(topk_ids[0].tolist()):
        local_expert = expert_id - expert_offset
        if not 0 <= local_expert < num_local_experts:
            continue
        gate_up = _w8a16_linear_npu(
            x,
            w13_weight[local_expert],
            w13_weight_scale[local_expert],
        )
        hidden_act = torch_npu.npu_swiglu(gate_up)
        routed = _w8a16_linear_npu(
            hidden_act,
            w2_weight[local_expert],
            w2_weight_scale[local_expert],
        )
        route_weight = topk_weights[0, route_index].to(x.dtype)
        route_outputs[route_index].copy_((routed * route_weight).to(torch.float32).squeeze(0))
    return route_outputs.sum(dim=0, keepdim=True).to(x.dtype)


def _w8a8_packed_grouped_experts_npu(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    expert_offset: int,
) -> torch.Tensor:
    """Run both expert GEMMs with cumulative group boundaries.

    Peer-owned routes sort after the local experts, so the last group boundary
    may be smaller than the route count. CANN leaves those output rows
    uninitialized. For prefill, one count sync lets us omit those rows from
    both GEMMs and all following elementwise work. Small batches keep the
    device-only path.
    """
    import torch_npu

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    if num_tokens <= _FUSED_ROUTING_MAX_TOKENS and x.device.type == "npu":
        # The 310P routing kernel computes the expert permutation, inverse
        # permutation, and cumulative group boundaries together. Its expert
        # range handling misses part of nonzero TP shards on this CANN build,
        # so present local IDs and put peer routes in the trailing sentinel.
        num_local_experts = w13_weight.shape[0]
        local_ids = topk_ids.to(torch.int32)
        if expert_offset:
            local_ids = torch.where(
                (local_ids >= expert_offset) & (local_ids < expert_offset + num_local_experts),
                local_ids - expert_offset,
                num_local_experts,
            )
        sorted_x, inverse_order, group_list, _ = torch_npu.npu_moe_init_routing_v2(
            x,
            local_ids,
            active_num=num_tokens * top_k,
            expert_num=num_local_experts,
            drop_pad_mode=0,
            active_expert_range=[0, num_local_experts],
            quant_mode=-1,
            row_idx_type=0,
        )
        # QuantGroupedMatmulDequant requires int64 cumulative group ends;
        # index_select accepts the routing kernel's int32 inverse directly.
        group_list = group_list.to(torch.int64)
        local_rows = torch.arange(num_tokens * top_k, device=x.device) < group_list[-1]
        gate_up = torch_npu.npu_quant_grouped_matmul_dequant(sorted_x, w13_weight, w13_weight_scale, group_list)
        gate_up = torch.where(local_rows[:, None], gate_up, 0)
        hidden_act = torch_npu.npu_swiglu(gate_up)
        routed = torch_npu.npu_quant_grouped_matmul_dequant(hidden_act, w2_weight, w2_weight_scale, group_list)
        routed = torch.where(local_rows[:, None], routed, 0)
        # Applying weights after unpermutation avoids gathering a separate
        # sorted weight vector, while keeping the old fp16 weight rounding.
        route_outputs = routed.to(torch.float32).index_select(0, inverse_order)
        route_outputs = route_outputs * topk_weights.to(x.dtype).reshape(-1, 1)
        return route_outputs.view(num_tokens, top_k, hidden).sum(dim=1).to(x.dtype)

    dispatch = build_grouped_expert_dispatch(
        topk_weights,
        topk_ids,
        num_local_experts=w13_weight.shape[0],
        expert_offset=expert_offset,
        weight_dtype=x.dtype,
    )
    sorted_order = dispatch.order
    group_list = dispatch.group_list
    if num_tokens >= _PACKED_LOCAL_ROUTE_MIN_TOKENS:
        # The group count is much smaller than the route tensor. At TP4 only
        # roughly a quarter of the 10 routes per token belong to this rank;
        # truncating here avoids full-sized gate, activation, and output
        # tensors for peer-owned routes. This single sync per prefill layer
        # is paid only for large batches and benchmarked with real weights.
        local_count = int(group_list[-1].item())
        if local_count == 0:
            return torch.zeros_like(x)
        local_order = sorted_order[:local_count]
        sorted_x = x[dispatch.token_indices[local_order]]
        gate_up = torch_npu.npu_quant_grouped_matmul_dequant(sorted_x, w13_weight, w13_weight_scale, group_list)
        hidden_act = torch_npu.npu_swiglu(gate_up)
        routed = torch_npu.npu_quant_grouped_matmul_dequant(hidden_act, w2_weight, w2_weight_scale, group_list)
        routed = routed.to(torch.float32) * dispatch.route_weights[local_order]
        # Every peer-owned slot maps to one trailing zero row. The inverse
        # permutation remains device-resident, including its sentinel clamp.
        local_and_zero = torch.cat((routed, routed.new_zeros((1, hidden))), dim=0)
        route_outputs = local_and_zero[dispatch.inverse_order.clamp_max(local_count)]
        return route_outputs.view(num_tokens, top_k, hidden).sum(dim=1).to(x.dtype)

    sorted_x = x[dispatch.token_indices[sorted_order]]
    local_rows = torch.arange(num_tokens * top_k, device=x.device) < group_list[-1]
    gate_up = torch_npu.npu_quant_grouped_matmul_dequant(sorted_x, w13_weight, w13_weight_scale, group_list)
    gate_up = torch.where(local_rows[:, None], gate_up, 0)
    hidden_act = torch_npu.npu_swiglu(gate_up)
    routed = torch_npu.npu_quant_grouped_matmul_dequant(hidden_act, w2_weight, w2_weight_scale, group_list)
    routed = torch.where(local_rows[:, None], routed, 0)
    routed = routed.to(torch.float32) * dispatch.route_weights[sorted_order]
    route_outputs = routed[dispatch.inverse_order]
    return route_outputs.view(num_tokens, top_k, hidden).sum(dim=1).to(x.dtype)


def w8a8_grouped_experts_npu(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    *,
    expert_offset: int = 0,
    use_packed_grouped: bool = False,
) -> torch.Tensor:
    """Use device-side grouped W8A8 for packed banks, or W8A16 fallback.

    The fallback copies a compact count vector to the host for prefill and the
    selected ids for decode. The packed prefill path reads only its local route
    count so its much larger peer-owned rows need not be materialized.
    """
    import torch_npu

    num_tokens, hidden = x.shape
    if use_packed_grouped:
        return _w8a8_packed_grouped_experts_npu(
            x,
            topk_weights,
            topk_ids,
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            expert_offset,
        )
    num_local_experts = len(w13_weight)
    if num_tokens == 1:
        return _w8a16_decode_experts_npu(
            x,
            topk_weights,
            topk_ids,
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            expert_offset,
        )
    top_k = topk_ids.shape[1]
    dispatch = build_grouped_expert_dispatch(
        topk_weights,
        topk_ids,
        num_local_experts=num_local_experts,
        expert_offset=expert_offset,
        weight_dtype=x.dtype,
    )
    # torch.bincount sizes its output from the maximum input value, even with
    # ``minlength``. On 310P a bad/asynchronous GatherV2 result here can request
    # an exabyte-sized allocation. Count only the known local expert ids from
    # the unsorted, already-clamped routes; the result is always fixed-size.
    counts = dispatch.counts
    # The sentinel bin sorts last. Once the existing count-vector sync completes,
    # gather only locally owned pairs instead of materializing peer-owned rows.
    local_counts = counts.tolist()
    local_order = dispatch.order[: sum(local_counts)]
    sorted_token = dispatch.token_indices[local_order]
    sorted_weight = dispatch.route_weights[local_order]
    sorted_x = x[sorted_token]
    # Keep expert outputs contiguous in sorted order. On 310P, index_copy_ into
    # the original route slots is extremely slow (hundreds of ms for a prefill
    # layer). An inverse-permutation gather restores the exact top-k slot order.
    # Zero peer-owned sentinel rows so this rank still contributes only its
    # local partial sum before the caller's all-reduce.
    local_count = sum(local_counts)
    # Only local routes need storage; one final zero row stands in for every
    # peer-owned route after the inverse permutation.
    sorted_outputs = torch.empty(local_count + 1, hidden, dtype=torch.float32, device=x.device)
    sorted_outputs[local_count].zero_()

    # This is one intentional device-to-host boundary per MoE layer.  It lets
    # decode invoke only the locally active top-k experts (typically 2-3 at
    # TP4) instead of launching 128 empty expert matmuls.  Each active expert
    # then uses the 310P-supported weight-only FP16-activation path.
    start = 0
    for expert_id, count in enumerate(local_counts):
        if count == 0:
            continue
        stop = start + count
        group_x = sorted_x[start:stop]
        gate_up = _w8a16_linear_npu(
            group_x,
            w13_weight[expert_id],
            w13_weight_scale[expert_id],
        )
        hidden_act = torch_npu.npu_swiglu(gate_up)
        routed = _w8a16_linear_npu(
            hidden_act,
            w2_weight[expert_id],
            w2_weight_scale[expert_id],
        )
        routed = routed * sorted_weight[start:stop]
        sorted_outputs[start:stop].copy_(routed.to(torch.float32))
        start = stop
    # Float32 route indices sort on the 310P AiCore. Fall back to integer sort
    # if a future token budget exceeds float32's exact-integer range.
    route_outputs = sorted_outputs[dispatch.inverse_order.clamp_max(local_count)]
    return route_outputs.view(num_tokens, top_k, hidden).sum(dim=1).to(x.dtype)


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
    *,
    expert_offset: int = 0,
    num_global_experts: int | None = None,
    use_packed_grouped: bool = False,
) -> torch.Tensor:
    """Grouped routed-expert W8A8 forward (``group_list`` grouped-matmul mimic).

    Every (token, slot) pair is sorted by its selected expert id; each expert
    evaluates its whole group in one QDQ MLP call (gmm1 -> swiglu -> gmm2), the
    result is scaled by the router weight and scattered back with ``index_add_``.
    Empty groups (skew) are skipped. All arithmetic is float32; the caller casts.

    Expert-dimension TP slicing (expert parallel): ``topk_ids`` always carry
    *global* expert ids (the router gate is replicated, so every rank selects
    identically and deterministically). When this rank's bank holds only a slice
    of the global expert set, pass ``expert_offset`` (its first global id) and
    ``num_global_experts``: ids outside ``[expert_offset, expert_offset +
    num_local)`` are routed into a sentinel bin and dropped without any extra
    synchronization (the single ``counts.tolist()`` boundary sync of the original
    path is preserved), and the returned tensor is this rank's *partial* sum --
    the caller all-reduces across ranks and adds the (replicated) shared expert.
    With the defaults (or ``num_global_experts == num_local``) this is exactly
    the unsharded full-bank forward.

    Args:
        x: ``[T, hidden]`` activations (float; upcast to float32 internally).
        topk_weights: ``[T, top_k]`` router weights (already renormalized/scaled).
        topk_ids: ``[T, top_k]`` selected *global* expert ids.
        w13_weight: ``[E_local, 2*moe, hidden]`` int8 fused gate/up weights.
        w13_weight_scale / w13_weight_offset: ``[E_local, 2*moe, 1]`` per-channel params.
        w2_weight: ``[E_local, hidden, moe]`` int8 down weights.
        w2_weight_scale / w2_weight_offset: ``[E_local, hidden, 1]`` per-channel params.
        expert_offset: first global expert id held locally (0 = full bank).
        num_global_experts: global expert count when sliced; ``None`` or equal
            to the local count means unsharded.

    Returns:
        ``[T, hidden]`` float32 routed-expert partial output (full output when
        unsharded; the caller all-reduces when TP-sliced).
    """
    if x.device.type == "npu":
        return w8a8_grouped_experts_npu(
            x,
            topk_weights,
            topk_ids,
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            expert_offset=expert_offset,
            use_packed_grouped=use_packed_grouped,
        )

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    num_experts = len(w13_weight)
    sharded = num_global_experts is not None and num_global_experts != num_experts
    x32 = x.to(torch.float32)

    pair_expert = topk_ids.reshape(-1)  # [T * top_k]
    if sharded:
        # Translate to local ids; ids owned by peer ranks land in a sentinel
        # bin (== num_experts) sorted last and dropped by the counts slice.
        pair_expert = pair_expert - expert_offset
        in_local = (pair_expert >= 0) & (pair_expert < num_experts)
        pair_expert = torch.where(in_local, pair_expert, torch.full_like(pair_expert, num_experts))
    pair_weight = topk_weights.to(torch.float32).reshape(-1, 1)
    pair_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    pair_x = x32[pair_token]  # gather activations for each (token, slot) pair

    # int64 argsort runs on AiCpu (slow + logs a warning); expert ids are < 2^24
    # so float32 preserves their exact ordering and keeps the sort on AiCore.
    order = torch.argsort(pair_expert.to(torch.float32), stable=True)
    sorted_expert = pair_expert[order]
    sorted_x = pair_x[order]
    sorted_weight = pair_weight[order]
    sorted_token = pair_token[order]

    counts = torch.bincount(sorted_expert, minlength=num_experts + (1 if sharded else 0))
    if sharded:
        counts = counts[:num_experts]  # drop the sentinel (peer-owned) bin
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
