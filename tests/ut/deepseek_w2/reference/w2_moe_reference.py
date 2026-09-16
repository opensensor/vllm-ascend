# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for the DeepSeek V4.1 2-bit (W2) expert MoE.

This is the core critical-path reference for the 552B-at-2-bit-experts harness.
It has two clearly separated pieces:

1. The **NEW** W2 packed-weight format: 2-bit signed codes plus one FP scale
   per source ``weight_block_size`` block (``[32, 32]``). Packing quantizes a
   float weight block to signed int2 codes ``{-2, -1, 0, 1}`` (four codes per
   ``uint8``) with a per-block scale chosen so the round-trip error is bounded
   by half a quantization step. Unpacking widens the codes back to int8 and
   applies the per-block scale, recovering an int8-domain weight the grouped
   MoE math then consumes.

2. The grouped INT8 dynamic quantize/dequantize (QDQ) MoE forward, whose
   activation path mirrors the established Ascend 310P W8A8 reference
   (``tests/ut/qwen38_1m/reference/w8a8_reference.py``): activations are
   quantized per token to symmetric int8 (``amax / 127``) and dequantized
   before the matmul. Only the *weight source* differs (W2 unpack instead of a
   stored int8 checkpoint), so this file imports and reuses the W8A8 activation
   quantizer rather than re-deriving it.

DeepSeek MoE geometry modelled here: ``num_experts`` routed experts (384 in the
production config), top-``top_k`` routing (6), and one always-on shared expert.
Each expert is a SwiGLU MLP (``down(silu(gate(x)) * up(x))``) whose three
linears store their weights in the W2 format.

The two-bit grid is intentionally the *signed* two's-complement int2 range
``{-2, -1, 0, 1}`` (the honest on-device 2-bit storage). Because that grid is
asymmetric, the per-block scale is chosen from both tails so that every weight
lands inside the nearest-rounding interval ``[-2.5, 1.5] * scale`` and the
reconstruction error stays within ``scale / 2`` per element.
"""

from __future__ import annotations

import torch

from tests.ut.qwen38_1m.reference.w8a8_reference import (
    dequantize_per_token,
    quantize_per_token_int8,
)

# --- W2 grid constants -----------------------------------------------------
# Signed two's-complement int2 codes.
W2_CODE_MIN = -2
W2_CODE_MAX = 1
# Codes are stored 4-per-byte; the sign bit is bit 1 of each 2-bit field.
W2_CODES_PER_BYTE = 4
W2_FIELD_MASK = 0b11
W2_SIGN_WRAP = 1 << 2  # 4: subtract to sign-extend a 2-bit field
# Nearest-rounding half-widths of the asymmetric grid: a weight rounds to a
# valid code (no clip beyond half a step) iff it lies in
# ``[W2_CODE_MIN - 0.5, W2_CODE_MAX + 0.5] * scale``. Choosing the scale from
# both tails with these divisors keeps every element inside that interval.
_W2_POS_COVER = W2_CODE_MAX + 0.5  # 1.5
_W2_NEG_COVER = -(W2_CODE_MIN - 0.5)  # 2.5
_W2_SCALE_FLOOR = 1e-12

# Source quant block (rows, cols) — DeepSeek V4.1 ``weight_block_size``.
W2_BLOCK_ROWS = 32
W2_BLOCK_COLS = 32


def _round_half_even(x: torch.Tensor) -> torch.Tensor:
    """Round-half-to-even, matching the asc int8 ``round_mode='rint'`` default."""
    return torch.round(x)


def compute_w2_block_scales(w: torch.Tensor) -> torch.Tensor:
    """Per-block scales for a weight matrix, one scalar per ``[32, 32]`` block.

    Args:
        w: ``[out, in]`` float weights; ``out`` and ``in`` are multiples of
            ``W2_BLOCK_ROWS`` / ``W2_BLOCK_COLS``.

    Returns:
        ``[out // 32, in // 32]`` float64 per-block scales. The scale of each
        block is ``max(pos_max / 1.5, neg_absmax / 2.5, floor)`` so that every
        element of the block rounds to a code within half a step.
    """
    w = w.double()
    out_features, in_features = w.shape
    if out_features % W2_BLOCK_ROWS or in_features % W2_BLOCK_COLS:
        raise ValueError(
            f"W2 weight shape {tuple(w.shape)} must tile [{W2_BLOCK_ROWS}, {W2_BLOCK_COLS}] blocks exactly."
        )
    blocks = w.view(
        out_features // W2_BLOCK_ROWS,
        W2_BLOCK_ROWS,
        in_features // W2_BLOCK_COLS,
        W2_BLOCK_COLS,
    ).permute(0, 2, 1, 3)  # [Br, Bc, 32, 32]
    pos_max = blocks.clamp_min(0).amax(dim=(-2, -1))
    neg_absmax = blocks.clamp_max(0).abs().amax(dim=(-2, -1))
    scale = torch.maximum(pos_max / _W2_POS_COVER, neg_absmax / _W2_NEG_COVER)
    return scale.clamp_min(_W2_SCALE_FLOOR)


def _broadcast_block_scales(block_scale: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Expand ``[Br, Bc]`` block scales to a full ``[out, in]`` grid."""
    return (
        block_scale.double()
        .repeat_interleave(W2_BLOCK_ROWS, dim=0)
        .repeat_interleave(W2_BLOCK_COLS, dim=1)[:out_features, :in_features]
    )


def quantize_weight_w2(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float weights to signed int2 codes + per-block scales.

    Args:
        w: ``[out, in]`` float weights.

    Returns:
        ``(codes, block_scale)`` with ``codes`` int8 in ``[-2, 1]`` shaped
        ``[out, in]`` and ``block_scale`` float64 ``[out // 32, in // 32]``.
    """
    w = w.double()
    out_features, in_features = w.shape
    block_scale = compute_w2_block_scales(w)
    full_scale = _broadcast_block_scales(block_scale, out_features, in_features)
    codes = _round_half_even(w / full_scale).clamp(W2_CODE_MIN, W2_CODE_MAX).to(torch.int8)
    return codes, block_scale


def pack_w2_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed int2 codes (last dim) into ``uint8``, four codes per byte.

    Args:
        codes: ``[..., in]`` int8 codes in ``[-2, 1]``; ``in`` a multiple of 4.

    Returns:
        ``[..., in // 4]`` uint8 packed codes (little-endian within a byte:
        code ``j`` occupies bits ``2j..2j+1``).
    """
    if codes.shape[-1] % W2_CODES_PER_BYTE:
        raise ValueError("last dim must be a multiple of 4 to pack W2 codes")
    nibble = (codes.to(torch.int64) & W2_FIELD_MASK).to(torch.uint8)
    groups = nibble.view(*nibble.shape[:-1], nibble.shape[-1] // W2_CODES_PER_BYTE, W2_CODES_PER_BYTE)
    packed = torch.zeros(groups.shape[:-1], dtype=torch.uint8)
    for j in range(W2_CODES_PER_BYTE):
        packed = packed | (groups[..., j] << (2 * j))
    return packed


def unpack_w2_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_w2_codes`; returns sign-extended int8 codes.

    Args:
        packed: ``[..., in // 4]`` uint8 packed codes.
        in_features: original last-dim length (``packed.shape[-1] * 4``).

    Returns:
        ``[..., in]`` int8 codes in ``[-2, 1]``.
    """
    if packed.shape[-1] * W2_CODES_PER_BYTE != in_features:
        raise ValueError("in_features does not match packed width")
    packed = packed.to(torch.int64)
    fields = []
    for j in range(W2_CODES_PER_BYTE):
        field = (packed >> (2 * j)) & W2_FIELD_MASK
        # Sign-extend: codes 2, 3 map to -2, -1.
        field = torch.where(field >= W2_SIGN_WRAP // 2, field - W2_SIGN_WRAP, field)
        fields.append(field)
    codes = torch.stack(fields, dim=-1).reshape(*packed.shape[:-1], in_features)
    return codes.to(torch.int8)


def unpack_w2_to_int8(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Unpack W2 → dequantized float weight (int8 codes × per-block scale).

    This is the "unpack W2 → INT8, apply per-block scale" step: the packed
    2-bit fields are widened to signed int8 codes and multiplied by their
    owning ``[32, 32]`` block scale, reproducing the float weight the MoE
    matmul consumes.

    Returns:
        ``[out, in]`` float64 dequantized weights.
    """
    codes = unpack_w2_codes(packed, in_features).double()
    full_scale = _broadcast_block_scales(block_scale, out_features, in_features)
    return codes * full_scale


def dequantize_weight_w2(
    codes: torch.Tensor,
    block_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize un-packed int8 codes with their per-block scale (no bit ops)."""
    out_features, in_features = codes.shape
    full_scale = _broadcast_block_scales(block_scale, out_features, in_features)
    return codes.double() * full_scale


def w2_qdq_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """One W2-weight linear with per-token int8 activation QDQ.

    ``y = dequant(quant_per_token(x)) @ dequant_w2(packed, scale).T``. The
    activation quantize/dequantize is the shared W8A8 symmetric per-token path.

    Args:
        x: ``[T, in]`` float activations.
        packed: ``[out, in // 4]`` uint8 packed W2 codes.
        block_scale: ``[out // 32, in // 32]`` per-block scales.
        out_features, in_features: weight shape.

    Returns:
        ``[T, out]`` float64 output.
    """
    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a).double()
    w_deq = unpack_w2_to_int8(packed, block_scale, out_features, in_features)
    return x_deq @ w_deq.t()


class W2Expert:
    """A single SwiGLU MoE expert with W2-quantized gate/up/down weights."""

    def __init__(
        self,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> None:
        # gate_w, up_w: [inter, hidden]; down_w: [hidden, inter].
        self.hidden = gate_w.shape[1]
        self.inter = gate_w.shape[0]
        self.gate_packed, self.gate_scale = _pack_from_float(gate_w)
        self.up_packed, self.up_scale = _pack_from_float(up_w)
        self.down_packed, self.down_scale = _pack_from_float(down_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[T, hidden]`` → ``[T, hidden]`` float64 expert output."""
        gate = w2_qdq_linear(x, self.gate_packed, self.gate_scale, self.inter, self.hidden)
        up = w2_qdq_linear(x, self.up_packed, self.up_scale, self.inter, self.hidden)
        act = torch.nn.functional.silu(gate) * up
        return w2_qdq_linear(act, self.down_packed, self.down_scale, self.hidden, self.inter)


def _pack_from_float(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize + pack a float weight into ``(packed, block_scale)``."""
    codes, block_scale = quantize_weight_w2(w)
    return pack_w2_codes(codes), block_scale


def route_topk(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax-then-top-k router with renormalized weights.

    Args:
        router_logits: ``[T, num_experts]``.
        top_k: experts kept per token.

    Returns:
        ``(expert_ids, weights)`` each ``[T, top_k]``; ``weights`` sum to 1 per
        row. Ties broken by ascending expert id (``torch.topk`` is stable for
        the sorted=True path used here).
    """
    probs = torch.softmax(router_logits.double(), dim=-1)
    top_w, top_ids = torch.topk(probs, top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    return top_ids, top_w


def w2_moe_forward(
    x: torch.Tensor,
    experts: list[W2Expert],
    router_logits: torch.Tensor,
    top_k: int,
    shared_expert: W2Expert | None = None,
) -> torch.Tensor:
    """Grouped W2 MoE forward: route, run each expert on its token batch, combine.

    Tokens routed to the same expert are processed together (the grouped path),
    matching how a batched device kernel dispatches. Because the activation
    quantizer is per-token, batching changes nothing numerically.

    Args:
        x: ``[T, hidden]`` float activations.
        experts: routed experts, indexed by expert id.
        router_logits: ``[T, num_experts]``.
        top_k: experts per token.
        shared_expert: optional always-on expert added to every token.

    Returns:
        ``[T, hidden]`` float64 output.
    """
    x = x.double()
    num_tokens, hidden = x.shape
    expert_ids, weights = route_topk(router_logits, top_k)
    out = torch.zeros(num_tokens, hidden, dtype=torch.float64)

    for expert_id, expert in enumerate(experts):
        # Rows (token, slot) routed to this expert.
        hit = expert_ids == expert_id
        if not hit.any():
            continue
        token_idx, slot_idx = torch.where(hit)
        batch = x[token_idx]
        expert_out = expert.forward(batch)
        scaled = expert_out * weights[token_idx, slot_idx].unsqueeze(-1)
        out.index_add_(0, token_idx, scaled)

    if shared_expert is not None:
        out = out + shared_expert.forward(x)
    return out


def w2_moe_forward_per_token(
    x: torch.Tensor,
    experts: list[W2Expert],
    router_logits: torch.Tensor,
    top_k: int,
    shared_expert: W2Expert | None = None,
) -> torch.Tensor:
    """Per-token reference: loop every token through its own experts.

    Independent of :func:`w2_moe_forward`'s grouping, so it is the oracle the
    grouped path is validated against.
    """
    x = x.double()
    num_tokens, hidden = x.shape
    expert_ids, weights = route_topk(router_logits, top_k)
    out = torch.zeros(num_tokens, hidden, dtype=torch.float64)

    for token in range(num_tokens):
        row = x[token : token + 1]  # [1, hidden]
        acc = torch.zeros(1, hidden, dtype=torch.float64)
        for slot in range(top_k):
            expert = experts[int(expert_ids[token, slot].item())]
            acc = acc + expert.forward(row) * weights[token, slot].item()
        if shared_expert is not None:
            acc = acc + shared_expert.forward(row)
        out[token] = acc.squeeze(0)
    return out


__all__ = [
    "W2_CODE_MIN",
    "W2_CODE_MAX",
    "W2_BLOCK_ROWS",
    "W2_BLOCK_COLS",
    "compute_w2_block_scales",
    "quantize_weight_w2",
    "pack_w2_codes",
    "unpack_w2_codes",
    "unpack_w2_to_int8",
    "dequantize_weight_w2",
    "w2_qdq_linear",
    "W2Expert",
    "route_topk",
    "w2_moe_forward",
    "w2_moe_forward_per_token",
]
