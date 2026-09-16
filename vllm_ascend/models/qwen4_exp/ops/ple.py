# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-free eager Qwen4Exp PLE gate + short-convolution ops (plan T4.3).

Ports the *formulas* (not the Triton kernels) from the vLLM CUDA fork's
``models/qwen4_exp/nvidia/ops/ple.py`` (``_ple_gate_kernel`` / ``_ple_conv_kernel``)
and ``nvidia/ple_layer.py`` to plain PyTorch so the PLE injection layer runs on
the host-only Ascend 310P dev path (no NPU, no Triton).

Gate (per token, per ``hc`` group of ``H`` lanes)::

    k_n = RMSNorm(key_group)    * (1 + norm_key_w)
    q_n = RMSNorm(hidden_group) * (1 + norm_query_w)
    d   = dot(k_n, q_n) / sqrt(H)
    g   = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))
    gated   = g * value                 # value is shared across the hc groups
    conv_in = RMSNorm(gated) * (1 + norm_conv_w)

Short convolution (dilated, causal, depthwise) is added into the gated output,
then the outer (hidden-state) residual::

    conv    = silu( sum_k weight[c, k] * conv_in[t - (K-1-k)*dilation, c] )
    output  = outer_residual + (gated + conv)

Precision policy: reductions (RMSNorm variance, the per-group dot product, and
the convolution accumulation) run in ``torch.promote_types(input_dtype,
accum_dtype)``. On the 310P device path the inputs are ``float16`` and
``accum_dtype`` is the policy's ``ple_norm_accumulation_dtype`` (``float32``),
so reductions accumulate in ``float32`` and the result rounds back to
``float16`` -- exactly the fork's fp32-accumulation split. When the CPU parity
harness feeds ``float64`` tensors the promotion keeps everything in ``float64``,
so the ops agree with the T0.6 ``ple_reference`` at rounding level.

This module deliberately re-derives the math independently of
``tests/ut/qwen38_1m/reference/ple_reference.py`` (the parity target); the gate
dot product is written as an ``einsum`` rather than an explicit lane sum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Magnitude floor applied before the gate sigmoid, matching the fork kernel's
# ``tl.maximum(tl.abs(d), 1e-6)`` and the T0.6 reference ``_GATE_MAG_FLOOR``.
GATE_MAGNITUDE_FLOOR = 1e-6

# Default accumulation dtype for reductions when a caller does not override it.
# The authoritative value is the dtype policy's ``ple_norm_accumulation_dtype``;
# callers (the PLE layer) pass it explicitly so no dtype literal is load-bearing.
_DEFAULT_ACCUM_DTYPE = torch.float32


def _compute_dtype(input_dtype: torch.dtype, accum_dtype: torch.dtype) -> torch.dtype:
    """Precision reductions run in: the wider of input and accumulation dtype."""
    return torch.promote_types(input_dtype, accum_dtype)


def ple_grouped_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    *,
    accum_dtype: torch.dtype = _DEFAULT_ACCUM_DTYPE,
) -> torch.Tensor:
    """RMSNorm applied independently to each contiguous group of ``group_size``.

    ``x`` is ``[T, C]`` with ``C`` a multiple of ``group_size``. Mirrors the
    fork's ``Qwen4ExpPLEGroupedNorm``: ``normalized * (1 + weight)``, with the
    variance reduction accumulated in the compute dtype and the result returned
    in that same (upcast) dtype so downstream reductions stay lossless.
    """
    if x.ndim != 2:
        raise ValueError("ple_grouped_rmsnorm expects a [T, C] tensor")
    seq_len, channels = x.shape
    if group_size <= 0 or channels % group_size:
        raise ValueError(f"channels ({channels}) must be a positive multiple of group_size ({group_size})")
    compute_dtype = _compute_dtype(x.dtype, accum_dtype)
    xc = x.to(compute_dtype)
    wc = weight.to(compute_dtype)
    grouped = xc.view(seq_len, channels // group_size, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).reshape(seq_len, channels)
    return normalized * (1.0 + wc)


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
    *,
    accum_dtype: torch.dtype = _DEFAULT_ACCUM_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PLE gated output and short-conv input.

    Args:
        key, hidden: ``[T, HC*H]`` (the projected key and the hc hidden state).
        value: ``[T, H]`` (shared across the ``HC`` groups).
        norm_key_w, norm_query_w, norm_conv_w: ``[HC*H]`` grouped-norm weights.
        eps: RMSNorm epsilon.
        accum_dtype: reduction accumulation dtype (policy fp32 on device).

    Returns:
        ``(gated, conv_input)`` each ``[T, HC*H]`` in the compute dtype.
    """
    if hidden.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("ple_gate expects 2-D key/value/hidden tensors")
    seq_len, hc_hidden = hidden.shape
    if key.shape != hidden.shape:
        raise ValueError(f"key {tuple(key.shape)} and hidden {tuple(hidden.shape)} must match")
    h = value.shape[-1]
    if value.shape[0] != seq_len:
        raise ValueError("value must share the token dimension with hidden")
    if h <= 0 or hc_hidden % h:
        raise ValueError(f"hc_hidden ({hc_hidden}) must be a positive multiple of H ({h})")
    hc = hc_hidden // h

    compute_dtype = _compute_dtype(
        _compute_dtype(_compute_dtype(key.dtype, value.dtype), hidden.dtype),
        accum_dtype,
    )
    value_c = value.to(compute_dtype)

    k_n = ple_grouped_rmsnorm(key, norm_key_w, eps, h, accum_dtype=compute_dtype)
    q_n = ple_grouped_rmsnorm(hidden, norm_query_w, eps, h, accum_dtype=compute_dtype)

    # Per-group dot product over the H lanes, written as an einsum (independent
    # of the reference's explicit lane-wise product-then-sum).
    k_g = k_n.view(seq_len, hc, h)
    q_g = q_n.view(seq_len, hc, h)
    dot = torch.einsum("thd,thd->th", k_g, q_g)  # [T, HC]
    d = dot / (h**0.5)
    sign = torch.sign(d)  # sign(0) == 0, matching the kernel's where-chain
    magnitude = torch.sqrt(torch.clamp(d.abs(), min=GATE_MAGNITUDE_FLOOR))
    gate = torch.sigmoid(sign * magnitude)  # [T, HC]

    gated = (gate.unsqueeze(-1) * value_c.unsqueeze(1)).reshape(seq_len, hc_hidden)
    conv_input = ple_grouped_rmsnorm(gated, norm_conv_w, eps, h, accum_dtype=compute_dtype)
    return gated, conv_input


def ple_short_conv(
    conv_input: torch.Tensor,
    gated: torch.Tensor,
    outer_residual: torch.Tensor,
    conv_weight: torch.Tensor,
    dilation: int,
    *,
    activation: str | None = "silu",
    accum_dtype: torch.dtype = _DEFAULT_ACCUM_DTYPE,
) -> torch.Tensor:
    """Dilated causal depthwise short convolution added to the gated output.

    This is the full-sequence (stateless) form used on the CPU parity path: a
    single prefill with no carried convolution state. The stateful decode/spec
    KV-cache routing is a device concern handled in the model runner, not here.

    Args:
        conv_input: ``[T, C]`` normalized gate output (conv input).
        gated: ``[T, C]`` gated output (inner residual).
        outer_residual: ``[T, C]`` hidden-state residual.
        conv_weight: ``[C, K]`` depthwise filters.
        dilation: convolution dilation.
        activation: ``"silu"`` or ``None``.
        accum_dtype: convolution accumulation dtype (policy fp32 on device).

    Returns:
        ``[T, C]`` PLE output.
    """
    if conv_input.shape != gated.shape or conv_input.shape != outer_residual.shape:
        raise ValueError("conv_input, gated, and outer_residual must share shape")
    if conv_weight.ndim != 2:
        raise ValueError("conv_weight must be [C, K]")
    seq_len, channels = conv_input.shape
    if conv_weight.shape[0] != channels:
        raise ValueError(f"conv_weight channels {conv_weight.shape[0]} != input channels {channels}")
    if dilation <= 0:
        raise ValueError("dilation must be positive")

    compute_dtype = _compute_dtype(
        _compute_dtype(_compute_dtype(conv_input.dtype, gated.dtype), outer_residual.dtype),
        accum_dtype,
    )
    conv_in_c = conv_input.to(compute_dtype)
    gated_c = gated.to(compute_dtype)
    outer_c = outer_residual.to(compute_dtype)
    weight_c = conv_weight.to(compute_dtype)

    kernel_size = weight_c.shape[-1]
    state_len = (kernel_size - 1) * dilation

    x_t = conv_in_c.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    x_pad = F.pad(x_t, (state_len, 0))
    conv = F.conv1d(
        x_pad,
        weight_c.unsqueeze(1),
        groups=channels,
        dilation=dilation,
    )
    conv = conv[..., :seq_len].squeeze(0).transpose(0, 1)  # [T, C]
    if activation == "silu":
        conv = conv * torch.sigmoid(conv)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation!r}")
    return outer_c + (gated_c + conv)


__all__ = [
    "GATE_MAGNITUDE_FLOOR",
    "ple_grouped_rmsnorm",
    "ple_gate",
    "ple_short_conv",
]
