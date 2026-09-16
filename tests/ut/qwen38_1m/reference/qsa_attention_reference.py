# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for Qwen4Exp QSA sparse attention.

Ports the *formulas* (not Triton) from:
  * ``vllm/models/qwen4_exp/nvidia/qsa.py`` +
    ``vllm/model_executor/models/qwen3_next.py::_project_qkv_gate`` -- the
    Q/K GemmaRMSNorm, partial RoPE, and the ``out * sigmoid(gate)`` output gate.
  * ``vllm/models/qwen4_exp/nvidia/ops/qsa.py``
    (``_qsa_sparse_paged_gqa_splitk_kernel``) -- softmax attention over the
    *selected* token set with GQA head grouping and softmax scale
    ``head_dim ** -0.5``.

The reference computes sparse attention by gathering the selected tokens; the
self-consistency oracle is dense attention masked to exactly the same set.
"""

from __future__ import annotations

import torch

# QSA sparse-attention geometry from the plan (T6.4): 24 query heads, 2 KV heads,
# head dim 256.
QSA_NUM_QUERY_HEADS = 24
QSA_NUM_KV_HEADS = 2
QSA_HEAD_DIM = 256

_NEG_INF = float("-inf")


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """GemmaRMSNorm over the last dim: ``x * rsqrt(mean(x^2)+eps) * (1+weight)``.

    Computed in float64 to match the model's float upcast.
    """
    x = x.double()
    weight = weight.double()
    variance = x.square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized * (1.0 + weight)


def apply_partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply neox-style RoPE to the first ``rotary_dim`` dims; pass the rest through.

    Args:
        x: ``[T, H, D]``.
        positions: ``[T]`` integer positions.
        rotary_dim: number of leading dims that are rotated (<= D, even).

    Returns:
        ``[T, H, D]``.
    """
    x = x.double()
    seq_len, num_heads, head_dim = x.shape
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and <= head_dim")
    positions = positions.double()
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    angles = positions[:, None] * inv_freq[None, :]  # [T, rotary_dim/2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # neox: duplicate halves so cos/sin apply to [x1, x2] with rotate_half.
    cos = torch.cat([cos, cos], dim=-1)[:, None, :]  # [T, 1, rotary_dim]
    sin = torch.cat([sin, sin], dim=-1)[:, None, :]

    rot = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    rot_out = rot * cos + rotate_half * sin
    return torch.cat([rot_out, passthrough], dim=-1)


def project_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    eps: float,
    rotary_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head Q/K GemmaRMSNorm then partial RoPE, in the model's order."""
    q = gemma_rmsnorm(q, q_norm_w, eps)
    k = gemma_rmsnorm(k, k_norm_w, eps)
    q = apply_partial_rope(q, positions, rotary_dim, base)
    k = apply_partial_rope(k, positions, rotary_dim, base)
    return q, k


def sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Gather-based sparse GQA attention with an output gate.

    Args:
        query: ``[T, Hq, D]``.
        key, value: ``[S, Hkv, D]`` full context caches.
        gate: ``[T, Hq, D]`` pre-sigmoid output gate.
        packed_indices: ``[T, W]`` selected token ids (-1 padded).
        valid_counts: ``[T]`` number of valid entries per row (the loop bound).
        num_kv_heads: KV head count for GQA grouping.

    Returns:
        ``[T, Hq, D]`` gated attention output.
    """
    query = query.double()
    key = key.double()
    value = value.double()
    gate = gate.double()
    seq_len, num_q_heads, head_dim = query.shape
    group_size = num_q_heads // num_kv_heads
    scale = head_dim**-0.5
    out = torch.zeros_like(query)

    for t in range(seq_len):
        count = int(valid_counts[t].item())
        idx = packed_indices[t, :count]
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        for h in range(num_q_heads):
            kv_head = h // group_size
            q_h = query[t, h]  # [D]
            k_sel = key[idx, kv_head]  # [n, D]
            v_sel = value[idx, kv_head]  # [n, D]
            scores = (k_sel @ q_h) * scale  # [n]
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs @ v_sel
    return out * torch.sigmoid(gate)


def dense_masked_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Dense oracle: full attention masked to exactly the selected set.

    Computes scores against *all* context tokens but sets non-selected logits to
    -inf before the softmax. Must equal :func:`sparse_gqa_attention`.
    """
    query = query.double()
    key = key.double()
    value = value.double()
    gate = gate.double()
    seq_len, num_q_heads, head_dim = query.shape
    context_len = key.shape[0]
    group_size = num_q_heads // num_kv_heads
    scale = head_dim**-0.5
    out = torch.zeros_like(query)

    for t in range(seq_len):
        count = int(valid_counts[t].item())
        idx = packed_indices[t, :count]
        idx = idx[idx >= 0]
        mask = torch.zeros(context_len, dtype=torch.bool)
        if idx.numel() > 0:
            mask[idx] = True
        if not mask.any():
            continue
        for h in range(num_q_heads):
            kv_head = h // group_size
            q_h = query[t, h]
            scores = (key[:, kv_head] @ q_h) * scale  # [S]
            scores = torch.where(mask, scores, torch.full_like(scores, _NEG_INF))
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs @ value[:, kv_head]
    return out * torch.sigmoid(gate)


__all__ = [
    "QSA_NUM_QUERY_HEADS",
    "QSA_NUM_KV_HEADS",
    "QSA_HEAD_DIM",
    "gemma_rmsnorm",
    "apply_partial_rope",
    "project_qk_norm_rope",
    "sparse_gqa_attention",
    "dense_masked_gqa_attention",
]
