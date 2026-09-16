# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P DeepSeek V4.1 Multi-head Latent Attention (MLA) -- eager (E3.1).

This is the *host-lane, Triton-free, torch_npu-free* MLA for the 310P DeepSeek
V4.1 W2 path. It ADAPTS the shipped device MLA rather than inventing new math:

Reused (structure / formulas), NOT re-derived
---------------------------------------------
* The absorbed-latent formulation is the exact one the shipped device impl
  computes -- ``vllm_ascend/attention/mla_v1.py::AscendMLAImpl``:
  ``_q_proj_and_k_up_proj`` folds the key up-projection into the query
  (``torch.bmm(q_nope, W_UK_T)``) and ``_v_up_proj`` folds the value
  up-projection into the output (``W_UV``). We reproduce that identity in eager
  torch here (``forward(..., absorbed=True)``) so the on-device kernel and this
  host module compute the same thing.
* The low-rank q down/up (``q_lora_rank``), the shared kv down-projection into a
  ``kv_lora_rank`` latent plus a decoupled ``qk_rope_head_dim`` RoPE tail, and
  the latent-KV write/read are the DeepSeek MLA geometry the shipped
  ``deepseek_v4`` ``DeepseekV4Attention`` / ``ops/mla.py`` wrapper wire on
  device. The E0.4 reference (``tests/ut/deepseek_w2/reference/mla_reference``)
  ports the same formulas; this module is the production twin, validated to
  match it at ``MLA_RTOL``/``MLA_ATOL``.

Eager-reimplemented (because the shipped path is device-only)
------------------------------------------------------------
* The shipped ``AscendMLAImpl`` / ``ops/mla.py`` / ``DeepseekV4Attention`` pull
  ``torch_npu`` grouped/absorbed MLA kernels, ``FusedMoEFactory`` and the DSA
  (deepseek-sparse-attention) sink/compressor/indexer stack -- none of which
  import on the CPU host (no NPU, no Triton). So the projections, RMSNorm,
  decoupled Neox RoPE, causal softmax attention and the latent-KV cache are
  plain eager torch here. On real 310P silicon E4.1 may swap the inner attend
  for the guarded ``npu_*`` op; the linears / RoPE / latent layout are
  unchanged.

Dtypes come from the E2.1 policy (:data:`ASCEND_DEEPSEEKV41_DTYPE_POLICY`):
FP16 main (``mla_dtype``), FP32 accumulation (``mla_accumulation_dtype``) for
RMSNorm / softmax / matmul reduction, and FP16 latent-KV cache
(``kv_cache_dtype``). No dtype literals are spelled here -- every dtype is read
from the policy. When the module is instantiated in float64 (parity harness)
the accumulation stays float64 so it matches the float64 reference exactly.

Interface E4.1 wires (via ``model.py``'s ``_override_mla_indexer`` hook; this
module never edits ``model.py``)::

    mla = AscendDeepseekV41MLA.from_config(text_config, dtype_policy=policy)
    out = mla(hidden_states, positions)  # prefill
    cache = mla.new_latent_kv_cache(max_seq_len)  # decode cache
    out_t = mla(hidden_t, positions_t, kv_cache=cache)  # incremental
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    DeepseekV41DtypePolicy,
)

# DeepSeek V4.1 decoupled-RoPE tail width and default RoPE base (theta). These
# mirror the E0.4 reference constants; the tail is never absorbed (not low-rank).
QK_ROPE_HEAD_DIM = 64
ROPE_BASE = 10000.0


# ---------------------------------------------------------------------------
# Eager building blocks (RMSNorm + decoupled Neox RoPE), formula-identical to
# the E0.4 reference so host parity holds at float64.
# ---------------------------------------------------------------------------
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, accum_dtype: torch.dtype) -> torch.Tensor:
    """``x * rsqrt(mean(x^2)+eps) * weight`` over the last dim, in ``accum_dtype``."""
    orig_dtype = x.dtype
    x = x.to(accum_dtype)
    w = weight.to(accum_dtype)
    variance = x.square().mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps) * w
    return normed.to(orig_dtype)


def apply_decoupled_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Neox-style RoPE over the last dim of ``x`` (``[..., D]``, D even).

    ``positions`` (shape ``[T]``) broadcasts over any middle head dims. The
    trig is always computed in float64 (like the E0.4 reference) then cast back
    to ``x``'s dtype, so the decoupled-tail numerics are formulation-invariant.
    """
    orig_dtype = x.dtype
    xd = x.double()
    rotary_dim = xd.shape[-1]
    if rotary_dim % 2:
        raise ValueError("rope dim must be even")
    pos = positions.double()
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64, device=x.device) / rotary_dim))
    angles = pos[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    while cos.ndim < xd.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    half = rotary_dim // 2
    x1 = xd[..., :half]
    x2 = xd[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return (xd * cos + rotate_half * sin).to(orig_dtype)


# ---------------------------------------------------------------------------
# Latent-KV cache: the compressed kv latent ``c`` [.., kv_lora_rank] plus the
# rotated decoupled RoPE key ``k_rope`` [.., qk_rope_head_dim]. This is exactly
# what the shipped device MLA persists per token (the absorbed form never
# materializes per-head K/V), sized/typed from the E2.1 policy.
# ---------------------------------------------------------------------------
@dataclass
class LatentKVCache:
    """Ring-free append cache of ``(latent c, rotated k_rope)`` per token."""

    c: torch.Tensor  # [max_seq_len, kv_lora_rank]
    k_rope: torch.Tensor  # [max_seq_len, qk_rope_head_dim]
    length: int = 0

    @property
    def kv_lora_rank(self) -> int:
        return self.c.shape[-1]

    @property
    def qk_rope_head_dim(self) -> int:
        return self.k_rope.shape[-1]

    def append(self, c_new: torch.Tensor, k_rope_new: torch.Tensor) -> None:
        """Write the current tokens' latent + rotated rope key into the cache."""
        n = c_new.shape[0]
        end = self.length + n
        if end > self.c.shape[0]:
            raise ValueError(f"latent-KV cache overflow: need {end}, capacity {self.c.shape[0]}")
        self.c[self.length : end] = c_new.to(self.c.dtype)
        self.k_rope[self.length : end] = k_rope_new.to(self.k_rope.dtype)
        self.length = end

    def valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the populated ``(c, k_rope)`` slices."""
        return self.c[: self.length], self.k_rope[: self.length]


class AscendDeepseekV41MLA(nn.Module):
    """Eager 310P DeepSeek V4.1 MLA (adapts the shipped device MLA formulas).

    Weight layout matches the E0.4 reference (and the shipped kv_b split into
    ``W_UK`` / ``W_UV``) so the host parity harness can load reference weights
    verbatim:

    * ``w_dq``     ``[q_lora_rank, hidden]``          -- q down-projection
    * ``q_a_norm`` ``[q_lora_rank]``                  -- RMSNorm on the q latent
    * ``w_uq``     ``[num_heads*qk_head_dim, q_lora]``-- q up-projection
    * ``w_dkv``    ``[kv_lora_rank+rope, hidden]``    -- shared kv down + rope key
    * ``kv_a_norm````[kv_lora_rank]``                 -- RMSNorm on the kv latent
    * ``w_uk``     ``[num_heads*qk_nope, kv_lora]``   -- key up (absorbed)
    * ``w_uv``     ``[num_heads*v_head, kv_lora]``    -- value up (absorbed)
    * ``w_o``      ``[hidden, num_heads*v_head]``     -- output projection
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
        v_head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_base: float = ROPE_BASE,
        dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dtype_policy = dtype_policy
        # Dtypes come from the policy (never spelled as literals). ``dtype``
        # overrides the compute/parameter dtype for the float64 parity harness.
        self.param_dtype = dtype if dtype is not None else dtype_policy.mla_dtype
        # float64 params keep float64 accumulation so we match the fp64 oracle.
        is_fp64 = self.param_dtype == torch.float64
        self.accum_dtype = self.param_dtype if is_fp64 else dtype_policy.mla_accumulation_dtype
        self.cache_dtype = self.param_dtype if is_fp64 else dtype_policy.kv_cache_dtype

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # DeepSeek softmax scale over the full (nope+rope) query dim.
        self.softmax_scale = self.qk_head_dim**-0.5
        self.eps = rms_norm_eps
        self.rope_base = rope_base

        H = self.num_heads
        factory = {"dtype": self.param_dtype, "device": device}
        self.w_dq = nn.Parameter(torch.empty(q_lora_rank, hidden_size, **factory))
        self.q_a_norm = nn.Parameter(torch.empty(q_lora_rank, **factory))
        self.w_uq = nn.Parameter(torch.empty(H * self.qk_head_dim, q_lora_rank, **factory))
        self.w_dkv = nn.Parameter(torch.empty(kv_lora_rank + qk_rope_head_dim, hidden_size, **factory))
        self.kv_a_norm = nn.Parameter(torch.empty(kv_lora_rank, **factory))
        self.w_uk = nn.Parameter(torch.empty(H * qk_nope_head_dim, kv_lora_rank, **factory))
        self.w_uv = nn.Parameter(torch.empty(H * self.v_head_dim, kv_lora_rank, **factory))
        self.w_o = nn.Parameter(torch.empty(hidden_size, H * self.v_head_dim, **factory))
        self.reset_parameters()

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> AscendDeepseekV41MLA:
        """Build from an HF-style DeepSeek V4.1 text config (E4.1 entry point)."""

        def g(name: str, default: object = None) -> object:
            return getattr(config, name, default)

        return cls(
            hidden_size=int(g("hidden_size")),
            num_attention_heads=int(g("num_attention_heads")),
            q_lora_rank=int(g("q_lora_rank")),
            kv_lora_rank=int(g("kv_lora_rank")),
            qk_nope_head_dim=int(g("qk_nope_head_dim")),
            qk_rope_head_dim=int(g("qk_rope_head_dim", QK_ROPE_HEAD_DIM)),
            v_head_dim=(int(g("v_head_dim")) if g("v_head_dim") is not None else None),
            rms_norm_eps=float(g("rms_norm_eps", 1e-6)),
            rope_base=float(g("rope_theta", ROPE_BASE)),
            dtype_policy=dtype_policy,
            dtype=dtype,
            device=device,
        )

    def reset_parameters(self, seed: int | None = None) -> None:
        """Deterministic small init (device-agnostic; real weights come from load)."""
        gen = torch.Generator().manual_seed(0 if seed is None else seed)

        def fill(p: nn.Parameter, scale: float, bias: float = 0.0) -> None:
            vals = torch.randn(p.shape, generator=gen, dtype=torch.float64) * scale + bias
            with torch.no_grad():
                p.copy_(vals.to(p.dtype))

        fill(self.w_dq, 0.05)
        fill(self.q_a_norm, 0.1, bias=1.0)
        fill(self.w_uq, 0.05)
        fill(self.w_dkv, 0.05)
        fill(self.kv_a_norm, 0.1, bias=1.0)
        fill(self.w_uk, 0.05)
        fill(self.w_uv, 0.05)
        fill(self.w_o, 0.05)

    # -- latent-KV cache ----------------------------------------------------
    def new_latent_kv_cache(
        self,
        max_seq_len: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> LatentKVCache:
        """Allocate a latent-KV cache sized/typed from config + policy.

        ``c`` is ``[max_seq_len, kv_lora_rank]`` and ``k_rope`` is
        ``[max_seq_len, qk_rope_head_dim]``; dtype defaults to the policy
        ``kv_cache_dtype`` (FP16) unless the module runs in float64.
        """
        cache_dtype = dtype if dtype is not None else self.cache_dtype
        return LatentKVCache(
            c=torch.zeros(max_seq_len, self.kv_lora_rank, dtype=cache_dtype, device=device),
            k_rope=torch.zeros(max_seq_len, self.qk_rope_head_dim, dtype=cache_dtype, device=device),
        )

    # -- projections --------------------------------------------------------
    def _project_q(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden -> (q_nope [T,H,nope], q_rope [T,H,rope]) after down/up + norm."""
        q_a = _rms_norm(hidden @ self.w_dq.t(), self.q_a_norm, self.eps, self.accum_dtype)
        q = (q_a @ self.w_uq.t()).view(-1, self.num_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]
        return q_nope, q_rope

    def project_latent_kv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden -> (latent c [T, kv_lora], k_rope [T, rope]) after down + norm.

        This is the latent-KV *write payload* (pre-RoPE key); the module rotates
        ``k_rope`` by position before it lands in the cache.
        """
        kv_a = hidden @ self.w_dkv.t()
        c = _rms_norm(kv_a[:, : self.kv_lora_rank], self.kv_a_norm, self.eps, self.accum_dtype)
        k_rope = kv_a[:, self.kv_lora_rank :]
        return c, k_rope

    # -- attention ----------------------------------------------------------
    def _causal_mask(self, num_q: int, num_kv: int, past_len: int, device: torch.device) -> torch.Tensor:
        """Additive ``[num_q, num_kv]`` mask; query ``i`` sees key ``j<=past_len+i``."""
        q_pos = torch.arange(num_q, device=device)[:, None] + past_len
        k_pos = torch.arange(num_kv, device=device)[None, :]
        mask = torch.zeros(num_q, num_kv, dtype=self.accum_dtype, device=device)
        mask.masked_fill_(k_pos > q_pos, float("-inf"))
        return mask

    def _attend_absorbed(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        c_all: torch.Tensor,
        k_rope_all: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Absorbed-latent attention (shipped device form): fold W_UK into the
        query and W_UV into the output; attend directly over the latent."""
        H, nope, kv = self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank
        w_uk = self.w_uk.view(H, nope, kv).to(self.accum_dtype)
        w_uv = self.w_uv.view(H, self.v_head_dim, kv).to(self.accum_dtype)
        q_nope = q_nope.to(self.accum_dtype)
        q_rope = q_rope.to(self.accum_dtype)
        c_all = c_all.to(self.accum_dtype)
        k_rope_all = k_rope_all.to(self.accum_dtype)
        # Absorb keys: qc = W_UK_h^T q_nope  (mirrors bmm(q_nope, W_UK_T)).
        qc = torch.einsum("thn,hnk->thk", q_nope, w_uk)  # [T,H,kv]
        score_nope = torch.einsum("thk,sk->ths", qc, c_all)  # [T,H,S]
        score_rope = torch.einsum("thr,sr->ths", q_rope, k_rope_all)  # [T,H,S]
        scores = (score_nope + score_rope) * self.softmax_scale + mask[:, None, :]
        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum("ths,sk->thk", probs, c_all)  # [T,H,kv]
        out = torch.einsum("thk,hvk->thv", context, w_uv)  # [T,H,v]
        return out

    def _attend_dense(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        c_all: torch.Tensor,
        k_rope_all: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dense oracle form: materialize per-head K/V from the latent, then MHA."""
        H, nope = self.num_heads, self.qk_nope_head_dim
        S = c_all.shape[0]
        c_all = c_all.to(self.accum_dtype)
        k_rope_all = k_rope_all.to(self.accum_dtype)
        k_nope = (c_all @ self.w_uk.t().to(self.accum_dtype)).view(S, H, nope)
        v = (c_all @ self.w_uv.t().to(self.accum_dtype)).view(S, H, self.v_head_dim)
        q_full = torch.cat([q_nope.to(self.accum_dtype), q_rope.to(self.accum_dtype)], dim=-1)  # [T,H,d]
        k_rope_exp = k_rope_all[:, None, :].expand(S, H, self.qk_rope_head_dim)
        k_full = torch.cat([k_nope, k_rope_exp], dim=-1)  # [S,H,d]
        scores = torch.einsum("thd,shd->ths", q_full, k_full) * self.softmax_scale + mask[:, None, :]
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("ths,shv->thv", probs, v)  # [T,H,v]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        kv_cache: LatentKVCache | None = None,
        absorbed: bool = True,
    ) -> torch.Tensor:
        """MLA forward.

        Args:
            hidden_states: ``[T, hidden]`` activations for the current step.
            positions: ``[T]`` integer positions (defaults to ``arange(T)`` when
                there is no cache; required for incremental decode).
            kv_cache: optional :class:`LatentKVCache` for incremental decode;
                current tokens are appended and the query attends over the whole
                cache (causal). ``None`` runs full-sequence prefill.
            absorbed: use the absorbed-latent form (device twin) when True, the
                dense oracle otherwise. Both are mathematically identical.

        Returns:
            ``[T, hidden]`` attention output.
        """
        num_q = hidden_states.shape[0]
        if positions is None:
            past_len = kv_cache.length if kv_cache is not None else 0
            positions = torch.arange(past_len, past_len + num_q, device=hidden_states.device)

        q_nope, q_rope = self._project_q(hidden_states)
        c, k_rope = self.project_latent_kv(hidden_states)
        q_rope = apply_decoupled_rope(q_rope, positions, self.rope_base)
        k_rope_rot = apply_decoupled_rope(k_rope, positions, self.rope_base)

        if kv_cache is not None:
            past_len = kv_cache.length
            kv_cache.append(c, k_rope_rot)
            c_all, k_rope_all = kv_cache.valid()
        else:
            past_len = 0
            c_all, k_rope_all = c, k_rope_rot

        mask = self._causal_mask(num_q, c_all.shape[0], past_len, hidden_states.device)
        attend = self._attend_absorbed if absorbed else self._attend_dense
        out = attend(q_nope, q_rope, c_all, k_rope_all, mask)  # [T,H,v]
        out = out.reshape(num_q, -1) @ self.w_o.t().to(self.accum_dtype)
        return out.to(hidden_states.dtype)


__all__ = [
    "QK_ROPE_HEAD_DIM",
    "ROPE_BASE",
    "AscendDeepseekV41MLA",
    "LatentKVCache",
    "apply_decoupled_rope",
]
