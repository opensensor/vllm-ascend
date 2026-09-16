# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for DeepSeek V4.1 Multi-head Latent Attention (MLA).

Ports the *formulas* (never the fused/Triton kernels) from the fork
``vllm/models/deepseek_v41/attention.py`` and ``nvidia/model.py``: the low-rank
query down/up projection (``q_lora_rank`` 1280), the KV down/up projection into
a shared latent, decoupled RoPE on the ``qk_rope_head_dim`` (64) tail, and
latent-KV attention.

Two mathematically equivalent formulations are provided and cross-checked:

* :func:`mla_dense_reference` materializes per-head keys/values from the latent
  and runs ordinary causal multi-head attention. This is the readable oracle.
* :func:`mla_absorbed_reference` folds the key up-projection into the query and
  the value up-projection into the output, so attention runs directly over the
  compressed latent (``kv_lora_rank`` dims) plus the decoupled RoPE tail — the
  form the on-device sparse-MLA kernel actually computes.

The absorption identity is exact:
``q_nope . (W_UK c_s) = (W_UK^T q_nope) . c_s`` and
``sum_s p_s (W_UV c_s) = W_UV (sum_s p_s c_s)``,
so the two agree up to float64 softmax/matmul reassociation only. The decoupled
RoPE tail is carried separately and never absorbed (it is not low-rank), matching
DeepSeek's decoupled rotary design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# DeepSeek V4.1 decoupled-RoPE tail width.
QK_ROPE_HEAD_DIM = 64
# Default RoPE base (theta).
ROPE_BASE = 10000.0


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain RMSNorm over the last dim (float64), ``x * rsqrt(mean(x^2)+eps) * w``."""
    x = x.double()
    weight = weight.double()
    variance = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Neox-style RoPE over the whole last dim of ``x`` (``[..., D]``, D even).

    ``positions`` broadcasts over the leading dims: shape ``[T]`` with ``x``
    ``[T, ..., D]``.
    """
    x = x.double()
    rotary_dim = x.shape[-1]
    if rotary_dim % 2:
        raise ValueError("rope dim must be even")
    positions = positions.double()
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    angles = positions[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    # Broadcast [T, D] over any middle head dims.
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    half = rotary_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate_half * sin


@dataclass
class MLAConfig:
    """MLA geometry (subset of the DeepSeek V4.1 attention config)."""

    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM
    v_head_dim: int = 0  # defaults to qk_nope_head_dim
    eps: float = 1e-6
    rope_base: float = ROPE_BASE

    def __post_init__(self) -> None:
        if self.v_head_dim == 0:
            self.v_head_dim = self.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # DeepSeek's softmax scale over the full (nope+rope) query dim.
        self.softmax_scale = self.qk_head_dim**-0.5


@dataclass
class MLAWeights:
    """MLA projection weights (all float64). Shapes in :func:`make_random_weights`."""

    w_dq: torch.Tensor  # [q_lora_rank, hidden]
    q_a_norm: torch.Tensor  # [q_lora_rank]
    w_uq: torch.Tensor  # [num_heads * qk_head_dim, q_lora_rank]
    w_dkv: torch.Tensor  # [kv_lora_rank + qk_rope_head_dim, hidden]
    kv_a_norm: torch.Tensor  # [kv_lora_rank]
    w_uk: torch.Tensor  # [num_heads * qk_nope_head_dim, kv_lora_rank]
    w_uv: torch.Tensor  # [num_heads * v_head_dim, kv_lora_rank]
    w_o: torch.Tensor  # [hidden, num_heads * v_head_dim]


def make_random_weights(cfg: MLAConfig, seed: int) -> MLAWeights:
    """Deterministic small random MLA weights for tests."""
    gen = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.05):
        return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale

    return MLAWeights(
        w_dq=r(cfg.q_lora_rank, cfg.hidden_size),
        q_a_norm=1.0 + r(cfg.q_lora_rank, scale=0.1),
        w_uq=r(cfg.num_heads * cfg.qk_head_dim, cfg.q_lora_rank),
        w_dkv=r(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        kv_a_norm=1.0 + r(cfg.kv_lora_rank, scale=0.1),
        w_uk=r(cfg.num_heads * cfg.qk_nope_head_dim, cfg.kv_lora_rank),
        w_uv=r(cfg.num_heads * cfg.v_head_dim, cfg.kv_lora_rank),
        w_o=r(cfg.hidden_size, cfg.num_heads * cfg.v_head_dim),
    )


def _project_q(hidden: torch.Tensor, cfg: MLAConfig, w: MLAWeights) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden -> (q_nope [T,H,nope], q_rope [T,H,rope]) after down/up + norm."""
    q_a = rms_norm(hidden @ w.w_dq.t(), w.q_a_norm, cfg.eps)
    q = (q_a @ w.w_uq.t()).view(-1, cfg.num_heads, cfg.qk_head_dim)
    q_nope = q[..., : cfg.qk_nope_head_dim]
    q_rope = q[..., cfg.qk_nope_head_dim :]
    return q_nope, q_rope


def _project_kv_latent(hidden: torch.Tensor, cfg: MLAConfig, w: MLAWeights) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden -> (latent c [T, kv_lora], k_rope [T, rope]) after down + norm."""
    kv_a = hidden @ w.w_dkv.t()
    c = rms_norm(kv_a[:, : cfg.kv_lora_rank], w.kv_a_norm, cfg.eps)
    k_rope = kv_a[:, cfg.kv_lora_rank :]
    return c, k_rope


def _causal_mask(seq_len: int) -> torch.Tensor:
    """``[T, T]`` additive mask: 0 where j<=i, -inf above the diagonal."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float64)
    mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def mla_dense_reference(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    cfg: MLAConfig,
    w: MLAWeights,
) -> torch.Tensor:
    """Dense MLA oracle: materialize per-head K/V from the latent, then MHA.

    Args:
        hidden: ``[T, hidden]`` float activations.
        positions: ``[T]`` integer positions (for RoPE).
        cfg, w: geometry and weights.

    Returns:
        ``[T, hidden]`` float64 attention output.
    """
    hidden = hidden.double()
    seq_len = hidden.shape[0]
    q_nope, q_rope = _project_q(hidden, cfg, w)
    c, k_rope = _project_kv_latent(hidden, cfg, w)

    k_nope = (c @ w.w_uk.t()).view(seq_len, cfg.num_heads, cfg.qk_nope_head_dim)
    v = (c @ w.w_uv.t()).view(seq_len, cfg.num_heads, cfg.v_head_dim)

    q_rope = apply_rope(q_rope, positions, cfg.rope_base)  # [T,H,rope]
    k_rope_rot = apply_rope(k_rope, positions, cfg.rope_base)  # [T,rope], shared

    mask = _causal_mask(seq_len)
    out = torch.zeros(seq_len, cfg.num_heads, cfg.v_head_dim, dtype=torch.float64)
    for h in range(cfg.num_heads):
        q_h = torch.cat([q_nope[:, h], q_rope[:, h]], dim=-1)  # [T, qk_head_dim]
        k_h = torch.cat([k_nope[:, h], k_rope_rot], dim=-1)  # [T, qk_head_dim]
        scores = (q_h @ k_h.t()) * cfg.softmax_scale + mask
        probs = torch.softmax(scores, dim=-1)
        out[:, h] = probs @ v[:, h]
    return out.reshape(seq_len, -1) @ w.w_o.t()


def mla_absorbed_reference(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    cfg: MLAConfig,
    w: MLAWeights,
) -> torch.Tensor:
    """Absorbed MLA: attend over the raw latent (no per-head K/V materialization).

    Folds ``W_UK`` into the query and ``W_UV`` into the output. Must equal
    :func:`mla_dense_reference`.
    """
    hidden = hidden.double()
    seq_len = hidden.shape[0]
    q_nope, q_rope = _project_q(hidden, cfg, w)
    c, k_rope = _project_kv_latent(hidden, cfg, w)

    q_rope = apply_rope(q_rope, positions, cfg.rope_base)
    k_rope_rot = apply_rope(k_rope, positions, cfg.rope_base)

    w_uk = w.w_uk.view(cfg.num_heads, cfg.qk_nope_head_dim, cfg.kv_lora_rank)
    w_uv = w.w_uv.view(cfg.num_heads, cfg.v_head_dim, cfg.kv_lora_rank)

    mask = _causal_mask(seq_len)
    out = torch.zeros(seq_len, cfg.num_heads, cfg.v_head_dim, dtype=torch.float64)
    for h in range(cfg.num_heads):
        # Absorb keys: q in latent space qc = W_UK_h^T q_nope.
        qc = q_nope[:, h] @ w_uk[h]  # [T, kv_lora]
        score_nope = qc @ c.t()  # [T, S]
        score_rope = q_rope[:, h] @ k_rope_rot.t()  # [T, S]
        scores = (score_nope + score_rope) * cfg.softmax_scale + mask
        probs = torch.softmax(scores, dim=-1)
        context_c = probs @ c  # [T, kv_lora]
        out[:, h] = context_c @ w_uv[h].t()  # W_UV_h (sum_s p c_s)
    return out.reshape(seq_len, -1) @ w.w_o.t()


__all__ = [
    "QK_ROPE_HEAD_DIM",
    "ROPE_BASE",
    "MLAConfig",
    "MLAWeights",
    "rms_norm",
    "apply_rope",
    "make_random_weights",
    "mla_dense_reference",
    "mla_absorbed_reference",
]
