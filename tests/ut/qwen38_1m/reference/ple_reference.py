# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for the Qwen4Exp PLE gate + short convolution.

Ports the *formulas* (not Triton) from ``vllm/models/qwen4_exp/nvidia/ops/ple.py``
(``_ple_gate_kernel`` and ``_ple_conv_kernel``) and ``ple_layer.py``.

Gate (per token, per hc group of ``H`` lanes)::

    k_n = RMSNorm(key_group)    * (1 + norm_key_w)
    q_n = RMSNorm(hidden_group) * (1 + norm_query_w)
    d   = dot(k_n, q_n) / sqrt(H)
    g   = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))
    gated  = g * value          # value is shared across hc groups
    conv_in = RMSNorm(gated) * (1 + norm_conv_w)

Short convolution (dilated, causal, depthwise) then adds into the gated output,
then the outer residual::

    conv    = silu( sum_k weight[c, k] * conv_in[t - (K-1-k)*dilation, c] )
    output  = outer_residual + (gated + conv)
"""

from __future__ import annotations

import torch

# Matches the kernel's magnitude floor before the sigmoid.
_GATE_MAG_FLOOR = 1e-6


def ple_grouped_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """RMSNorm applied independently to each contiguous group of ``group_size``.

    ``x`` is ``[T, C]`` with ``C`` a multiple of ``group_size``. Mirrors
    ``Qwen4ExpPLEGroupedNorm``: ``normalized * (1 + weight)`` in float.
    """
    x = x.double()
    weight = weight.double()
    seq_len, channels = x.shape
    grouped = x.view(seq_len, channels // group_size, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).reshape(seq_len, channels)
    return normalized * (1.0 + weight)


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the PLE gated output and conv input.

    Args:
        key, hidden: ``[T, HC*H]``.
        value: ``[T, H]`` (shared across the ``HC`` groups).
        norm_key_w, norm_query_w, norm_conv_w: ``[HC*H]`` norm weights.
        eps: RMSNorm epsilon.

    Returns:
        ``(gated, conv_input)`` each ``[T, HC*H]``.
    """
    key = key.double()
    value = value.double()
    hidden = hidden.double()
    seq_len, hc_hidden = hidden.shape
    h = value.shape[-1]
    hc = hc_hidden // h

    k_n = ple_grouped_rmsnorm(key, norm_key_w, eps, h)
    q_n = ple_grouped_rmsnorm(hidden, norm_query_w, eps, h)
    # Per-group dot product over H lanes.
    k_g = k_n.view(seq_len, hc, h)
    q_g = q_n.view(seq_len, hc, h)
    dot = (k_g * q_g).sum(dim=-1)  # [T, HC]
    d = dot / (h**0.5)
    sign = torch.sign(d)  # sign(0) == 0, matching the kernel's where-chain
    magnitude = torch.sqrt(torch.clamp(d.abs(), min=_GATE_MAG_FLOOR))
    g = torch.sigmoid(sign * magnitude)  # [T, HC]

    gated = (g.unsqueeze(-1) * value.unsqueeze(1)).reshape(seq_len, hc_hidden)
    conv_input = ple_grouped_rmsnorm(gated, norm_conv_w, eps, h)
    return gated, conv_input


def ple_short_conv(
    conv_input: torch.Tensor,
    gated: torch.Tensor,
    outer_residual: torch.Tensor,
    conv_weight: torch.Tensor,
    dilation: int,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Dilated causal depthwise short convolution added to the gated output.

    Args:
        conv_input: ``[T, C]`` normalized gate output (conv input).
        gated: ``[T, C]`` gated output (inner residual).
        outer_residual: ``[T, C]`` (the layer's hidden-state residual).
        conv_weight: ``[C, K]`` depthwise filters.
        dilation: convolution dilation.

    Returns:
        ``[T, C]`` PLE output.
    """
    conv_input = conv_input.double()
    gated = gated.double()
    outer_residual = outer_residual.double()
    conv_weight = conv_weight.double()
    seq_len, channels = conv_input.shape
    kernel_size = conv_weight.shape[-1]
    state_len = (kernel_size - 1) * dilation

    x_t = conv_input.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    x_pad = torch.nn.functional.pad(x_t, (state_len, 0))
    conv = torch.nn.functional.conv1d(
        x_pad,
        conv_weight.unsqueeze(1),
        groups=channels,
        dilation=dilation,
    )
    conv = conv[..., :seq_len].squeeze(0).transpose(0, 1)  # [T, C]
    if activation == "silu":
        conv = conv * torch.sigmoid(conv)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")
    return outer_residual + (gated + conv)


__all__ = [
    "ple_grouped_rmsnorm",
    "ple_gate",
    "ple_short_conv",
]
