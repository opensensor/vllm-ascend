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
decoder assembly to read.

End-to-end decoder-layer assembly (plan T6.4) is :func:`run_qsa_decoder_attention`:
the single composition -- project (Q/K/V/gate + indexer Q/K) -> indexer select ->
sparse GQA attention with the output gate -> out-projection -- that every one of
the model's 12 QSA layers runs. The full-model assembly's ``_QSAAttention`` owns
the projection weights, the indexer and the attention module, and delegates its
forward to this function so all QSA layers exercise identical code. It reproduces
the T0.6 composite reference (``qsa_indexer_reference`` + ``qsa_attention_reference``
composed) at rounding level under a float64 policy (see
``tests/ut/qwen38_1m/test_qsa_decoder_e2e.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
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


def _mrope_interleaved_dims(section: list[int]) -> list[int]:
    """Match vLLM's Qwen MRoPE frequency-to-axis interleaving."""
    if len(section) != 3 or any(size <= 0 for size in section):
        raise ValueError("mrope_section must contain three positive sizes")
    remaining = {axis: size for axis, size in enumerate(section)}
    remaining[0] -= 1
    original = remaining.copy()
    placed = {axis: 0 for axis in remaining}
    result: list[int] = []
    previous = None
    for _ in range(sum(remaining.values())):
        candidates = [axis for axis, count in remaining.items() if count > 0 and axis != previous]
        if not candidates:
            candidates = [axis for axis, count in remaining.items() if count > 0]
        axis = min(candidates, key=lambda item: (placed[item] / original[item], item))
        result.append(axis)
        placed[axis] += 1
        remaining[axis] -= 1
        previous = axis
    result.append(0)
    return result


def _rope_frequency_positions(
    positions: torch.Tensor,
    rotary_dim: int,
    mrope_section: list[int] | None,
    mrope_interleaved: bool,
) -> torch.Tensor:
    """Return one position per token and RoPE frequency pair."""
    half = rotary_dim // 2
    if positions.ndim == 1:
        return positions[:, None].expand(-1, half)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("positions must be [T] or multimodal [3,T]")
    if mrope_section is None:
        return positions[0, :, None].expand(-1, half)
    if not mrope_interleaved or sum(mrope_section) != half:
        raise ValueError("interleaved mrope_section must sum to rotary_dim // 2")
    axes = torch.tensor(_mrope_interleaved_dims(mrope_section), device=positions.device)
    return positions.index_select(0, axes).transpose(0, 1)


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
    *,
    mrope_section: list[int] | None = None,
    mrope_interleaved: bool = False,
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
    pos = _rope_frequency_positions(
        positions,
        rotary_dim,
        mrope_section,
        mrope_interleaved,
    ).to(accum_dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=accum_dtype, device=x.device) / rotary_dim))
    angles = pos * inv_freq[None, :]  # [T, rotary_dim/2]
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


def partial_rope_cos_sin(
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    dtype: torch.dtype,
    mrope_section: list[int] | None = None,
    mrope_interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize Neox-style cos/sin rows for a dedicated device kernel."""
    compute_dtype = torch.float32
    pos = _rope_frequency_positions(
        positions,
        rotary_dim,
        mrope_section,
        mrope_interleaved,
    ).to(compute_dtype)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=compute_dtype, device=positions.device) / rotary_dim)
    )
    angles = pos * inv_freq[None, :]
    cos = torch.cat((torch.cos(angles), torch.cos(angles)), dim=-1)
    sin = torch.cat((torch.sin(angles), torch.sin(angles)), dim=-1)
    return cos.to(dtype), sin.to(dtype)


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
        num_query_heads: int | None = None,
        num_kv_heads: int | None = None,
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

        self.num_query_heads = (
            int(getattr(config, "num_attention_heads", _DEFAULT_NUM_QUERY_HEADS))
            if num_query_heads is None
            else num_query_heads
        )
        self.num_kv_heads = (
            int(getattr(config, "num_key_value_heads", _DEFAULT_NUM_KV_HEADS)) if num_kv_heads is None else num_kv_heads
        )
        self.head_dim = int(getattr(config, "head_dim", _DEFAULT_HEAD_DIM))
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("QSA query heads must be divisible by kv heads")
        self.group_size = self.num_query_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", _DEFAULT_PARTIAL_ROTARY_FACTOR))
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        if self.rotary_dim % 2:
            raise ValueError("partial rotary_dim must be even")
        # Real Qwen4Exp configs nest ``rope_theta`` under ``rope_parameters``
        # (checkpoint value 1e7); older/tiny configs carry a top-level key. Read
        # the top-level key first, then fall back to ``rope_parameters``.
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.rope_theta = float(
            getattr(config, "rope_theta", None) or rope_parameters.get("rope_theta") or _DEFAULT_ROPE_THETA
        )
        section = rope_parameters.get("mrope_section")
        self.mrope_section = list(section) if section is not None else None
        self.mrope_interleaved = bool(rope_parameters.get("mrope_interleaved", False))
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
        q = apply_partial_rope(
            q,
            q_positions,
            self.rotary_dim,
            self.rope_theta,
            accum_dtype,
            mrope_section=self.mrope_section,
            mrope_interleaved=self.mrope_interleaved,
        )
        k = apply_partial_rope(
            k,
            k_positions,
            self.rotary_dim,
            self.rope_theta,
            accum_dtype,
            mrope_section=self.mrope_section,
            mrope_interleaved=self.mrope_interleaved,
        )
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


@dataclass(frozen=True)
class QSADecoderProjections:
    """The seven projection weights of one QSA decoder layer (``[out, hidden]``).

    Held by the full-model assembly's ``_QSAAttention`` and handed to
    :func:`run_qsa_decoder_attention`; the composition never owns parameters, so
    the assembly's state dict is unchanged by routing through it.
    """

    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    gate_proj: torch.Tensor
    index_q_proj: torch.Tensor
    index_k_proj: torch.Tensor
    out_proj: torch.Tensor


def run_qsa_decoder_attention(
    block_input: torch.Tensor,
    positions: torch.Tensor,
    *,
    projections: QSADecoderProjections,
    indexer: nn.Module,
    attention: AscendQwen4ExpQSAAttention,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    store_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Full QSA decoder-layer attention path (plan T6.4).

    The single composition every QSA layer runs::

        project (Q/K/V/gate + indexer Q/K)
          -> indexer select (T6.1)
          -> Q/K GemmaRMSNorm + partial RoPE -> sparse GQA attention (T6.2)
             with the ``out * sigmoid(gate)`` output gate
          -> out-projection

    Projections and the out-projection run in ``compute_dtype`` (the policy
    accumulation dtype) regardless of the stored (fp16) weight dtype, then the
    indexer / attention inputs are cast to ``store_dtype`` (the QSA main dtype),
    mirroring the storage/accumulation split pinned in ``dtype_policy``.

    Args:
        block_input: ``[T, hidden]`` mixed block input for this layer.
        positions: ``[T]`` logical positions of the query tokens.
        projections: the layer's seven projection weights.
        indexer: the weight-free QSA indexer (T6.1); called
            ``indexer(index_q, index_k, positions)`` -> selection.
        attention: the QSA sparse attention module (T6.2).
        num_query_heads, num_kv_heads, head_dim: attention geometry.
        index_n_heads, index_head_dim: indexer geometry.
        store_dtype: QSA main (storage) dtype for indexer / attention inputs.
        compute_dtype: accumulation dtype for the (out-)projections.

    Returns:
        ``[T, hidden]`` attention block output in ``store_dtype``.
    """
    seq_len = block_input.shape[0]

    def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(compute_dtype), weight.to(compute_dtype))

    query = _linear(block_input, projections.q_proj).view(seq_len, num_query_heads, head_dim)
    key = _linear(block_input, projections.k_proj).view(seq_len, num_kv_heads, head_dim)
    value = _linear(block_input, projections.v_proj).view(seq_len, num_kv_heads, head_dim)
    gate = _linear(block_input, projections.gate_proj).view(seq_len, num_query_heads, head_dim)
    index_q = _linear(block_input, projections.index_q_proj).view(seq_len, index_n_heads, index_head_dim)
    index_k = _linear(block_input, projections.index_k_proj)

    # QSA selection is causal in the flattened token stream. Multimodal RoPE
    # positions carry temporal/height/width axes, but only the temporal axis is
    # meaningful for visibility and compressed-group accounting.
    logical_positions = positions if positions.ndim == 1 else positions[0]
    selection = indexer(
        index_q.to(store_dtype),
        index_k.to(store_dtype),
        logical_positions,
    )
    out = attention(
        query.to(store_dtype),
        key.to(store_dtype),
        value.to(store_dtype),
        gate.to(store_dtype),
        positions,
        selection.token_indices,
        selection.valid_counts,
        key_positions=positions,
    )
    out = out.reshape(seq_len, num_query_heads * head_dim)
    return _linear(out, projections.out_proj).to(store_dtype)


__all__ = [
    "AscendQwen4ExpQSAAttention",
    "QSADecoderProjections",
    "partial_rope_cos_sin",
    "run_qsa_decoder_attention",
]
