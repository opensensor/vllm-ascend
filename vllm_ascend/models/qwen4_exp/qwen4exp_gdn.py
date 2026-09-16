# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P Qwen4Exp Gated DeltaNet (GDN) wiring (plan T5.1).

This adapter wires the Qwen4Exp GDN layers (short conv + gated delta-rule
recurrence, for the hidden-2560 config) to the existing Triton-free 310P
``fla`` kernels in :mod:`vllm_ascend._310p.ops.fla`:

* :func:`~vllm_ascend._310p.ops.fla.fused_gdn_gating.fused_gdn_gating_pytorch`
  -- the sigmoid/softplus gating math.
* :func:`~vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule.fused_recurrent_gated_delta_rule_pytorch`
  -- the token-recurrent (decode / unchunked) delta rule.
* :func:`~vllm_ascend._310p.ops.fla.chunk_gated_delta_rule.chunk_gated_delta_rule_310`
  (AscendC, NPU) / ``chunk_gated_delta_rule_pytorch`` (host fallback)
  -- the chunk-parallel (prefill) delta rule.

Scope (T5.1): shapes, dtypes and the numerical wiring only. The GDN *state
lifecycle* across prefill/decode/preemption is a separate task (T5.2); this
module only exposes clean, side-effect-free hooks and never mutates a cache.

Precision note (why a dedicated eager path exists)
--------------------------------------------------
The stock ``fla`` PyTorch fallbacks are *float32-internal*: the recurrent
kernel allocates its state as ``torch.float32``, the chunk kernel casts every
input ``.to(torch.float32)``, and the gating helper computes in float32. That
is exactly right for the fp16-main 310P production path (per the T1.2 dtype
policy), but it caps host parity against the float64 T0.6 reference at
~1e-7 -- it cannot reach the declared GDN tolerances (rtol 1e-8 / atol 1e-9).

So this adapter provides a *dtype-honoring* eager delta-rule that implements
the identical math parameterised by ``compute_dtype``. At ``float32`` it
reproduces the stock ``fla`` kernels (verified in the T5.1 UT); at ``float64``
it meets the T0.6 tolerances. ``backend="fla_pytorch"`` still routes through
the real ``fla`` kernels so the wiring itself is exercised; ``backend="auto"``
picks the AscendC kernels on NPU and the eager path on host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# --- kernel-matched constants (see gdn_reference / fused_gdn_gating) --------
QWEN4EXP_GDN_L2NORM_EPS = 1e-6
QWEN4EXP_GDN_SOFTPLUS_BETA = 1.0
QWEN4EXP_GDN_SOFTPLUS_THRESHOLD = 20.0
QWEN4EXP_GDN_CHUNK_SIZE = 64

# Conv-state block layout: "DS" -> (dim, state_len), "SD" -> (state_len, dim).
# Mirrors ``MambaStateShapeCalculator._orient_conv_shape`` without importing the
# vLLM config machinery (the 310P default is dim-first / "DS").
GDNConvStateLayout = Literal["DS", "SD"]

# Where the delta-rule math runs. "auto" -> AscendC kernels on NPU, else the
# host eager path. "fla_pytorch" -> the Triton-free fla PyTorch fallbacks.
GDNBackend = Literal["auto", "eager", "fla_pytorch", "ascend_npu"]

__all__ = [
    "QWEN4EXP_GDN_CHUNK_SIZE",
    "QWEN4EXP_GDN_L2NORM_EPS",
    "QWEN4EXP_GDN_SOFTPLUS_BETA",
    "QWEN4EXP_GDN_SOFTPLUS_THRESHOLD",
    "Qwen4ExpGDNParams",
    "gdn_conv_state_shape",
    "gdn_delta_rule",
    "gdn_gating",
    "gdn_recurrent_state_shape",
    "gdn_short_conv",
    "gdn_state_dtypes",
]


def _divide(numerator: int, denominator: int) -> int:
    if denominator <= 0 or numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}.")
    return numerator // denominator


@dataclass(frozen=True)
class Qwen4ExpGDNParams:
    """Config-derived GDN geometry for the Qwen4Exp (hidden-2560) linear layers.

    Field names mirror the HF ``linear_*`` config keys consumed by
    ``QwenGatedDeltaNetAttention``. ``head_dim`` / ``partial_rotary_factor`` are
    the *dense-attention* rotary params, carried here only so the short-conv and
    partial-rotary config can be validated from one place (GDN itself applies no
    rotary).
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    head_dim: int
    partial_rotary_factor: float

    @property
    def key_dim(self) -> int:
        return self.head_k_dim * self.num_k_heads

    @property
    def value_dim(self) -> int:
        return self.head_v_dim * self.num_v_heads

    @property
    def conv_dim(self) -> int:
        # mixed_qkv = [q(key_dim), k(key_dim), v(value_dim)] -> depthwise conv.
        return self.key_dim * 2 + self.value_dim

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def validate(self) -> None:
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "num_v_heads must be a multiple of num_k_heads "
                f"(grouped-value attention), got {self.num_v_heads} / {self.num_k_heads}."
            )
        if self.conv_kernel_size < 2:
            raise ValueError(f"GDN short-conv kernel must be >= 2, got {self.conv_kernel_size}.")
        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise ValueError(f"partial_rotary_factor must be in (0, 1], got {self.partial_rotary_factor}.")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError(f"partial rotary dim {self.rotary_dim} must be in (0, head_dim={self.head_dim}].")

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> Qwen4ExpGDNParams:
        """Build params from an HF Qwen4Exp text config.

        Reads the ``linear_*`` GDN keys plus ``head_dim`` /
        ``partial_rotary_factor`` (defaults match ``Qwen3NextConfig``).
        """
        params = cls(
            num_k_heads=int(config.linear_num_key_heads),
            num_v_heads=int(config.linear_num_value_heads),
            head_k_dim=int(config.linear_key_head_dim),
            head_v_dim=int(config.linear_value_head_dim),
            conv_kernel_size=int(config.linear_conv_kernel_dim),
            head_dim=int(getattr(config, "head_dim", 256)),
            partial_rotary_factor=float(getattr(config, "partial_rotary_factor", 0.25)),
        )
        params.validate()
        return params


# ---------------------------------------------------------------------------
# State shapes / dtypes (config- and policy-derived).
# ---------------------------------------------------------------------------
def gdn_conv_state_shape(
    params: Qwen4ExpGDNParams,
    tp_size: int = 1,
    num_spec: int = 0,
    layout: GDNConvStateLayout = "DS",
) -> tuple[int, int]:
    """Per-rank GDN short-conv cache shape.

    Mirrors ``MambaStateShapeCalculator.gated_delta_net_state_shape`` (conv
    part): ``dim = conv_dim / tp`` and ``state_len = conv_kernel - 1 + num_spec``.
    """
    conv_dim = _divide(params.conv_dim, tp_size)
    state_len = params.conv_kernel_size - 1 + num_spec
    if layout == "DS":
        return (conv_dim, state_len)
    return (state_len, conv_dim)


def gdn_recurrent_state_shape(
    params: Qwen4ExpGDNParams,
    tp_size: int = 1,
) -> tuple[int, int, int]:
    """Per-rank GDN recurrent (SSM) state shape ``[num_v_heads/tp, V, K]``."""
    return (
        _divide(params.num_v_heads, tp_size),
        params.head_v_dim,
        params.head_k_dim,
    )


def gdn_state_dtypes(
    policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
) -> tuple[torch.dtype, torch.dtype]:
    """Return ``(conv_cache_dtype, ssm_cache_dtype)`` from the T1.2 policy.

    Per policy: conv state rides the fp16 main dtype, the recurrent SSM state is
    kept in fp32. (The AscendC decode op currently narrows the SSM state to
    fp16 at the call boundary -- that is an op limitation handled in the state
    lifecycle task, not the config-derived expectation reported here.)
    """
    return (policy.mamba_conv_cache_dtype, policy.mamba_ssm_cache_dtype)


# ---------------------------------------------------------------------------
# Numerical wiring.
# ---------------------------------------------------------------------------
def _l2norm(x: torch.Tensor, eps: float = QWEN4EXP_GDN_L2NORM_EPS) -> torch.Tensor:
    """L2-normalize the last dim (host-safe twin of ``l2norm_310p``)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _resolve_backend(backend: GDNBackend, device: torch.device) -> GDNBackend:
    if backend != "auto":
        return backend
    return "ascend_npu" if device.type == "npu" else "eager"


def gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    backend: GDNBackend = "eager",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(token, head) log-decay ``g`` and gate ``beta`` for the delta rule.

    Args:
        A_log, dt_bias: ``[H]`` per-head params.
        a, b: ``[T, H]`` selective projections.
        compute_dtype: math dtype for the eager path (default: policy accum).
        backend: ``"fla_pytorch"`` routes through
            ``fused_gdn_gating_pytorch``; ``"eager"`` runs the identical
            formula in ``compute_dtype``.

    Returns:
        ``(g, beta)`` each ``[T, H]``; ``g`` is the (non-positive) log decay.
    """
    if backend == "fla_pytorch":
        from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch

        g, beta = fused_gdn_gating_pytorch(
            A_log,
            a,
            b,
            dt_bias,
            beta=QWEN4EXP_GDN_SOFTPLUS_BETA,
            threshold=QWEN4EXP_GDN_SOFTPLUS_THRESHOLD,
        )
        # fla returns a leading (seq-block) dim; drop it to match [T, H].
        return g.squeeze(0), beta.squeeze(0)

    dtype = compute_dtype or ASCEND_QWEN4EXP_DTYPE_POLICY.accumulation_dtype
    A_log_f = A_log.to(dtype)
    dt_bias_f = dt_bias.to(dtype)
    a_f = a.to(dtype)
    b_f = b.to(dtype)
    x = a_f + dt_bias_f
    beta_x = QWEN4EXP_GDN_SOFTPLUS_BETA * x
    softplus_x = torch.where(
        beta_x <= QWEN4EXP_GDN_SOFTPLUS_THRESHOLD,
        (1.0 / QWEN4EXP_GDN_SOFTPLUS_BETA) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    g = -torch.exp(A_log_f) * softplus_x
    beta = torch.sigmoid(b_f)
    return g, beta


def gdn_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Causal depthwise 1D short conv over the ``mixed_qkv`` stream.

    Left-pads by ``kernel_size - 1`` (output length == input length), one filter
    per channel, optional SiLU. Host twin of the AscendC ``npu_causal_conv1d_310``
    path; matches the T0.6 ``causal_depthwise_conv1d`` reference.

    Args:
        x: ``[T, C]`` sequence-major activations (``C == conv_dim``).
        weight: ``[C, kernel_size]`` depthwise filters.
    """
    dtype = compute_dtype or ASCEND_QWEN4EXP_DTYPE_POLICY.mamba_conv_cache_dtype
    x = x.to(dtype)
    weight = weight.to(dtype)
    seq_len, channels = x.shape
    kernel_size = weight.shape[-1]
    x_t = x.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    x_pad = torch.nn.functional.pad(x_t, (kernel_size - 1, 0))
    out = torch.nn.functional.conv1d(
        x_pad,
        weight.unsqueeze(1),
        bias=bias.to(dtype) if bias is not None else None,
        groups=channels,
    )
    out = out[..., :seq_len].squeeze(0).transpose(0, 1)  # [T, C]
    if activation == "silu":
        out = out * torch.sigmoid(out)
    elif activation is not None:
        raise ValueError(f"Unsupported GDN conv activation: {activation!r}")
    return out


def _expand_kv_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    """Expand ``[T, Hk, D]`` q/k to ``[T, Hv, D]`` (grouped-value attention)."""
    h = x.shape[1]
    if h == num_v_heads:
        return x
    return x.repeat_interleave(_divide(num_v_heads, h), dim=1)


def _delta_rule_eager_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
    use_qk_l2norm: bool,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-recurrent gated delta rule; dtype-honoring twin of the fla kernel."""
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)
    g = g.to(compute_dtype)
    beta = beta.to(compute_dtype)
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    q = q * scale
    num_v_heads = v.shape[1]
    q = _expand_kv_heads(q, num_v_heads)
    k = _expand_kv_heads(k, num_v_heads)

    seq_len, _, k_dim = q.shape
    v_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(num_v_heads, v_dim, k_dim, dtype=compute_dtype, device=q.device)
    else:
        state = initial_state.to(compute_dtype).clone()

    out = torch.empty(seq_len, num_v_heads, v_dim, dtype=compute_dtype, device=q.device)
    for t in range(seq_len):
        state = state * torch.exp(g[t]).view(num_v_heads, 1, 1)
        u = v[t] - torch.sum(state * k[t].unsqueeze(-2), dim=-1)
        u = u * beta[t].view(num_v_heads, 1)
        state = state + u.unsqueeze(-1) * k[t].unsqueeze(-2)
        out[t] = torch.sum(state * q[t].unsqueeze(-2), dim=-1)
    return out, state


def _delta_rule_eager_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
    use_qk_l2norm: bool,
    chunk_size: int,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel gated delta rule; dtype-honoring twin of the fla chunk kernel.

    Mirrors the WY/UT block form used by ``chunk_gated_delta_rule`` (the
    Transformers ``torch_chunk_gated_delta_rule`` flow) on a single sequence.
    """
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)
    g = g.to(compute_dtype)
    beta = beta.to(compute_dtype)
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    num_v_heads = v.shape[1]
    q = _expand_kv_heads(q, num_v_heads)
    k = _expand_kv_heads(k, num_v_heads)

    seq_len, _, k_dim = q.shape
    v_dim = v.shape[-1]

    # [T, H, D] -> [1, H, T, D]; [T, H] -> [1, H, T]
    query = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    key = k.permute(1, 0, 2).unsqueeze(0).contiguous()
    value = v.permute(1, 0, 2).unsqueeze(0).contiguous()
    beta_h = beta.permute(1, 0).unsqueeze(0).contiguous()
    g_h = g.permute(1, 0).unsqueeze(0).contiguous()

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = torch.nn.functional.pad(query, (0, 0, 0, pad))
    key = torch.nn.functional.pad(key, (0, 0, 0, pad))
    value = torch.nn.functional.pad(value, (0, 0, 0, pad))
    beta_h = torch.nn.functional.pad(beta_h, (0, pad))
    g_h = torch.nn.functional.pad(g_h, (0, pad))
    total_len = seq_len + pad
    query = query * scale

    v_beta = value * beta_h.unsqueeze(-1)
    k_beta = key * beta_h.unsqueeze(-1)

    def _to_chunks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])

    query, key, value, k_beta, v_beta = (_to_chunks(x) for x in (query, key, value, k_beta, v_beta))
    g_h = g_h.reshape(g_h.shape[0], g_h.shape[1], -1, chunk_size)

    mask_diag = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    g_h = g_h.cumsum(dim=-1)
    decay_mask = ((g_h.unsqueeze(-1) - g_h.unsqueeze(-2)).tril().exp()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_h.exp().unsqueeze(-1))

    batch = query.shape[0]
    if initial_state is None:
        state = torch.zeros(batch, num_v_heads, v_dim, k_dim, dtype=compute_dtype, device=q.device)
    else:
        state = initial_state.to(compute_dtype).unsqueeze(0).clone()
    core_out = torch.zeros_like(value)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)

    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask_upper, 0)
        v_prime = k_cumdecay[:, :, i] @ state.transpose(-1, -2)
        v_new = v_i - v_prime
        inter = (q_i * g_h[:, :, i, :, None].exp()) @ state.transpose(-1, -2)
        core_out[:, :, i] = inter + attn_intra @ v_new
        state = state * g_h[:, :, i, -1, None, None].exp() + v_new.transpose(-1, -2) @ (
            k_i * (g_h[:, :, i, -1, None] - g_h[:, :, i]).exp()[..., None]
        )

    core_out = core_out.reshape(batch, num_v_heads, -1, v_dim)[:, :, :seq_len]
    out = core_out.transpose(1, 2).contiguous()[0]  # [T, H, V]
    return out, state[0]


def _delta_rule_fla_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
    use_qk_l2norm: bool,
    chunked: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route through the real Triton-free fla PyTorch kernels (float32-internal).

    Exercises the actual kernel wiring on host. L2-norm is applied here (the
    kernel's own ``l2norm_310p`` needs an NPU RMSNorm), so the kernels are
    called with ``use_qk_l2norm_in_kernel=False``.
    """
    num_v_heads = v.shape[1]
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    # [T, H, D] -> [B=1, T, H, D]; kernels apply scale = K**-0.5 internally.
    q4 = q.unsqueeze(0)
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    g3 = _expand_kv_heads(g.unsqueeze(-1), num_v_heads).squeeze(-1).unsqueeze(0)
    beta3 = _expand_kv_heads(beta.unsqueeze(-1), num_v_heads).squeeze(-1).unsqueeze(0)
    init = None if initial_state is None else initial_state.unsqueeze(0)

    if chunked:
        from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch

        out, state = chunk_gated_delta_rule_pytorch(
            q4,
            k4,
            v4,
            g3,
            beta3,
            scale=scale,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        return out.squeeze(0), state.squeeze(0)

    from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import (
        fused_recurrent_gated_delta_rule_pytorch,
    )

    out, state = fused_recurrent_gated_delta_rule_pytorch(
        q4,
        k4,
        v4,
        g3,
        beta3,
        initial_state=init,
        use_qk_l2norm_in_kernel=False,
    )
    return out.squeeze(0), state.squeeze(0)


def gdn_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    chunked: bool = False,
    chunk_size: int = QWEN4EXP_GDN_CHUNK_SIZE,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    compute_dtype: torch.dtype | None = None,
    backend: GDNBackend = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta-rule attention over one sequence.

    Args:
        q, k: ``[T, Hk, K]`` raw (pre-l2norm) query/key.
        v: ``[T, Hv, V]``. ``Hv`` may be a multiple of ``Hk`` (grouped value).
        g, beta: ``[T, Hv]`` from :func:`gdn_gating`.
        initial_state: ``[Hv, V, K]`` recurrent state, or ``None`` (zeros).
        chunked: chunk-parallel (prefill) form vs. token-recurrent (decode).
        scale: query scale; defaults to ``K ** -0.5`` (kernel default).
        use_qk_l2norm: apply L2 norm to q/k before scaling (kernel semantics).
        compute_dtype: eager math dtype (default: policy accumulation dtype).
        backend: ``"auto"`` (NPU->AscendC, host->eager), ``"eager"``,
            ``"fla_pytorch"``, or ``"ascend_npu"``.

    Returns:
        ``(o, final_state)`` with ``o`` ``[T, Hv, V]`` and state ``[Hv, V, K]``.
    """
    resolved = _resolve_backend(backend, q.device)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if resolved == "ascend_npu":
        return _delta_rule_ascend_npu(q, k, v, g, beta, initial_state, scale, use_qk_l2norm, chunked)
    if resolved == "fla_pytorch":
        return _delta_rule_fla_pytorch(q, k, v, g, beta, initial_state, scale, use_qk_l2norm, chunked)

    dtype = compute_dtype or ASCEND_QWEN4EXP_DTYPE_POLICY.accumulation_dtype
    if chunked:
        return _delta_rule_eager_chunked(q, k, v, g, beta, initial_state, scale, use_qk_l2norm, chunk_size, dtype)
    return _delta_rule_eager_recurrent(q, k, v, g, beta, initial_state, scale, use_qk_l2norm, dtype)


def _delta_rule_ascend_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
    use_qk_l2norm: bool,
    chunked: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AscendC-backed GDN path (NPU only).

    This is the production 310P entry point. The full prefill/decode metadata
    plumbing (varlen ``cu_seqlens``, ``ssm_state_indices``) lives in the GDN
    state-lifecycle task (T5.2); here we only bridge single-sequence tensors to
    the ``chunk_gated_delta_rule_310`` kernel for prefill.
    """
    if chunked:
        from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310

        num_v_heads = v.shape[1]
        g3 = _expand_kv_heads(g.unsqueeze(-1), num_v_heads).squeeze(-1).unsqueeze(0)
        beta3 = _expand_kv_heads(beta.unsqueeze(-1), num_v_heads).squeeze(-1).unsqueeze(0)
        init = None if initial_state is None else initial_state.unsqueeze(0)
        out, state = chunk_gated_delta_rule_310(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g3,
            beta3,
            scale=scale,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        return out.squeeze(0), state.squeeze(0)

    raise NotImplementedError(
        "AscendC recurrent (decode) GDN wiring requires the T5.2 state-lifecycle "
        "metadata (cu_seqlens / ssm_state_indices); use backend='eager' on host."
    )
