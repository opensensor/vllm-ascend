# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp QSA (query-sparse attention) on the 310P host path (plan T6.2).

Fills the T1.2 stub with the torch-eager QSA sparse attention kernel: per query
token it applies per-head Q/K GemmaRMSNorm + partial (neox) RoPE, gathers the
indexer-selected ``<=2,048`` context rows (plus the causal tail of the open
group) from the main K/V caches through the T6.1 gather, runs GQA softmax
attention over exactly the selected set with an ``out * sigmoid(gate)`` output
gate, and honors the packed selection-count semantics (zero-selection first
token -> zero output).

Geometry (plan T6.4): 24 query heads, 2 KV heads, head dim 256, partial rotary
factor 0.25 (rotary_dim 64). Dtypes are pinned by
:data:`ASCEND_QWEN4EXP_DTYPE_POLICY` -- storage in ``qsa_main_dtype`` (float16
baseline, BF16-equivalent) and score/softmax accumulation in
``attention_accumulation_dtype`` (float32). No dtype literals are spelled here.

The math is numerically equivalent to the T0.6 eager reference
(``tests/ut/qwen38_1m/reference/qsa_attention_reference.py``): running with a
float64 accumulation reproduces it within the declared QSA attention tolerance.
No Triton/CUDA kernel is imported on the 310P path; the inner tile loop is a
clean seam a fused NPU kernel replaces later.

Chunked prefill (T6.3) is wired through :mod:`.ops.qsa_cache` (the MRV2
torch-fallback metadata builder advances the ring / compressed side caches one
chunk at a time, bit-identically to a whole pass); the chunk-size knob and the
preemption-aware recompute policy live in :mod:`.chunk_config` and are attached
here as :attr:`AscendQwen4ExpQSAAttention.chunk_prefill_policy` for the T6.4
decoder assembly to read. End-to-end decoder assembly (T6.4) is out of scope;
this module exposes the projection + attention seams they wire together.
"""

from __future__ import annotations

import torch
from torch import nn

from .chunk_config import QSAChunkPrefillPolicy
from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy
from .ops.qsa_attention import (
    QSAKVQuantHook,
    qsa_sparse_gqa_attention,
    qsa_write_kv_to_cache,
)

# QSA sparse-attention geometry defaults (plan T6.4), mirroring the T0.6
# reference: 24 query heads, 2 KV heads, head dim 256.
_DEFAULT_NUM_QUERY_HEADS = 24
_DEFAULT_NUM_KV_HEADS = 2
_DEFAULT_HEAD_DIM = 256
# Partial rotary: rotary_dim == head_dim * partial_rotary_factor (256 * 0.25 = 64).
_DEFAULT_PARTIAL_ROTARY_FACTOR = 0.25
_DEFAULT_ROPE_THETA = 10_000.0
_DEFAULT_RMS_NORM_EPS = 1e-6


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, accum_dtype: torch.dtype) -> torch.Tensor:
    """Per-head GemmaRMSNorm over the last dim: ``x * rsqrt(mean(x^2)+eps) * (1+w)``.

    Computed in ``accum_dtype`` (float32 on the 310P path; float64 reproduces the
    T0.6 reference exactly).
    """
    orig_dtype = x.dtype
    x = x.to(accum_dtype)
    w = weight.to(accum_dtype)
    variance = x.square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return (normalized * (1.0 + w)).to(orig_dtype)


def apply_partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    base: float,
    accum_dtype: torch.dtype,
) -> torch.Tensor:
    """Neox-style RoPE on the first ``rotary_dim`` dims; pass the rest through.

    Args:
        x: ``[T, H, D]``.
        positions: ``[T]`` integer positions.
        rotary_dim: leading dims that are rotated (even, ``<= D``).
        base: RoPE theta.
        accum_dtype: rotation compute dtype.

    Returns:
        ``[T, H, D]`` rotated tensor (in ``x``'s original dtype).
    """
    if rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and <= head_dim")
    orig_dtype = x.dtype
    x = x.to(accum_dtype)
    pos = positions.to(accum_dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=accum_dtype, device=x.device) / rotary_dim))
    angles = pos[:, None] * inv_freq[None, :]  # [T, rotary_dim/2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)[:, None, :]
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)[:, None, :]

    rot = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    rot_out = rot * cos + rotate_half * sin
    return torch.cat([rot_out, passthrough], dim=-1).to(orig_dtype)


class AscendQwen4ExpQSAAttention(nn.Module):
    """Query-sparse attention (torch-eager, 310P host path).

    Runs storage in ``policy.qsa_main_dtype`` with score accumulation in
    ``policy.attention_accumulation_dtype``. The K/V cache write path reserves a
    quant hook for Candidate A / C8 (task T8.1) via :attr:`kv_write_quant_hook`;
    it is unset here (float16/BF16 write path).
    """

    def __init__(
        self,
        *,
        config: object,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        chunk_prefill_policy: QSAChunkPrefillPolicy | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.dtype_policy = dtype_policy
        # T6.3 chunked-prefill knob + preemption-aware recompute policy. Prefill
        # runs the ring / compressed side caches one chunk at a time via
        # ``ops.qsa_cache.run_qsa_prefill``; this layer's ``forward`` is chunk
        # agnostic (it attends over whatever context it is handed).
        self.chunk_prefill_policy = chunk_prefill_policy or QSAChunkPrefillPolicy()
        self.qsa_dtype = dtype_policy.cast_site("qsa")
        self.kv_cache_dtype = dtype_policy.cast_site("kv_cache")
        self.accumulation_dtype = dtype_policy.cast_site("attention_accumulation")

        self.num_query_heads = int(getattr(config, "num_attention_heads", _DEFAULT_NUM_QUERY_HEADS))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", _DEFAULT_NUM_KV_HEADS))
        self.head_dim = int(getattr(config, "head_dim", _DEFAULT_HEAD_DIM))
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("QSA query heads must be divisible by kv heads")
        self.group_size = self.num_query_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", _DEFAULT_PARTIAL_ROTARY_FACTOR))
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        if self.rotary_dim % 2:
            raise ValueError("partial rotary_dim must be even")
        self.rope_theta = float(getattr(config, "rope_theta", _DEFAULT_ROPE_THETA))
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", _DEFAULT_RMS_NORM_EPS))

        # Per-head GemmaRMSNorm weights (applied as ``1 + weight``); zero-init is
        # the identity so an unloaded module is a no-op norm.
        self.q_norm_weight = nn.Parameter(torch.zeros(self.head_dim, dtype=self.qsa_dtype))
        self.k_norm_weight = nn.Parameter(torch.zeros(self.head_dim, dtype=self.qsa_dtype))

        # C8 write-path quant hook (Candidate A / task T8.1): reserved, unset.
        self.kv_write_quant_hook: QSAKVQuantHook | None = None

    def project_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        *,
        accum_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head Q/K GemmaRMSNorm then partial RoPE (model order)."""
        q = gemma_rmsnorm(query, self.q_norm_weight, self.rms_norm_eps, accum_dtype)
        k = gemma_rmsnorm(key, self.k_norm_weight, self.rms_norm_eps, accum_dtype)
        q = apply_partial_rope(q, q_positions, self.rotary_dim, self.rope_theta, accum_dtype)
        k = apply_partial_rope(k, k_positions, self.rotary_dim, self.rope_theta, accum_dtype)
        return q, k

    def write_kv_cache(
        self,
        key_cache_flat: torch.Tensor,
        value_cache_flat: torch.Tensor,
        slot_mapping: torch.Tensor,
        key_rows: torch.Tensor,
        value_rows: torch.Tensor,
    ) -> None:
        """Paged K/V write (BF16/float16), routing through the reserved C8 hook."""
        qsa_write_kv_to_cache(
            key_cache_flat,
            value_cache_flat,
            slot_mapping,
            key_rows,
            value_rows,
            quant_hook=self.kv_write_quant_hook,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        positions: torch.Tensor,
        packed_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        *,
        key_positions: torch.Tensor | None = None,
        accum_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Run QSA sparse attention for one sequence's query tokens.

        Args:
            query: ``[T, Hq, D]`` pre-norm queries.
            key, value: ``[S, Hkv, D]`` full-context K/V caches. ``key`` is
                pre-norm (normed + RoPE'd here); ``value`` is used as-is.
            gate: ``[T, Hq, D]`` pre-sigmoid output gate.
            positions: ``[T]`` logical positions of the query tokens.
            packed_indices: ``[T, W]`` indexer selection (``-1`` padded).
            valid_counts: ``[T]`` per-row valid count (the tile-loop bound).
            key_positions: ``[S]`` positions of the context keys; defaults to
                ``arange(S)`` (whole causal context).
            accum_dtype: accumulation dtype override; defaults to the policy's
                ``attention_accumulation_dtype``.

        Returns:
            ``[T, Hq, D]`` gated attention output cast to ``qsa_main_dtype``.
        """
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query/key/value must be [tokens, heads, head_dim]")
        accum = self.accumulation_dtype if accum_dtype is None else accum_dtype

        context_len = key.shape[0]
        if key_positions is None:
            key_positions = torch.arange(context_len, device=key.device)

        # Storage rides the QSA main dtype; norm/rope/attention accumulate in accum.
        q = query.to(self.qsa_dtype)
        k = key.to(self.qsa_dtype)
        v = value.to(self.qsa_dtype)
        gate_cast = gate.to(self.qsa_dtype)

        q, k = self.project_qk(q, k, positions, key_positions, accum_dtype=accum)

        out = qsa_sparse_gqa_attention(
            q,
            k,
            v,
            gate_cast,
            packed_indices,
            valid_counts,
            self.num_kv_heads,
            scale=self.scaling,
            accum_dtype=accum,
            apply_output_gate=True,
        )
        return out.to(self.qsa_dtype)


__all__ = ["AscendQwen4ExpQSAAttention"]
