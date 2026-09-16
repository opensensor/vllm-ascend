# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for W8A8 dynamic INT8 quantize/dequantize.

Ports the *math* (not the ``torch_npu`` kernels) from the Ascend 310P W8A8
dynamic scheme (``vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py`` and
peers) and the established asc int8 quant reference
(``torch.round(x / scale).clamp(-128, 127)``, per-token, ``round_mode="rint"``):

  * Activations: dynamic, *symmetric*, per-token INT8. The scale is
    ``amax(|x_row|) / 127`` and there is no activation offset (the scheme is
    symmetric on activations, per the in-source note).
  * Weights: per-output-channel INT8 loaded from the checkpoint with a
    ``weight_scale`` and ``weight_offset``. Offset application order matters:
    quantize is ``q = round(w / scale + offset)``; dequantize is the inverse
    ``w = (q - offset) * scale`` (matching the asc ``round(x*inv_scale+offset)``
    quantize / ``(q-offset)*scale`` dequant pairing).
"""

from __future__ import annotations

import torch

INT8_MIN = -128
INT8_MAX = 127
# Symmetric per-token grid uses the positive half-range (127 levels).
_SYM_LEVELS = 127.0


def _round_half_even(x: torch.Tensor) -> torch.Tensor:
    """Round-half-to-even (``rint``), matching the asc round_mode default."""
    return torch.round(x)  # torch.round is round-half-to-even


def quantize_per_token_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric per-token INT8 quantization of activations.

    Args:
        x: ``[T, C]`` float activations.

    Returns:
        ``(q, scale)`` with ``q`` int8 ``[T, C]`` and ``scale`` float ``[T, 1]``.
    """
    x = x.float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    # Rows that are all-zero get scale 1.0 to avoid div-by-zero; they quantize
    # to all-zero regardless.
    scale = torch.where(amax > 0, amax / _SYM_LEVELS, torch.ones_like(amax))
    q = _round_half_even(x / scale).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return q, scale


def dequantize_per_token(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_per_token_int8` (symmetric, no offset)."""
    return q.to(torch.float32) * scale


def compute_weight_qparams(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive per-output-channel asymmetric INT8 ``(scale, offset)`` for weights.

    ``w`` is ``[out_channels, in_channels]``. Mirrors an asymmetric checkpoint:
    ``scale = (wmax - wmin) / 255`` and ``offset`` (zero point) chosen so the
    dequant ``(q - offset) * scale`` reproduces the range.
    """
    w = w.float()
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    span = (wmax - wmin).clamp_min(1e-12)
    scale = span / (INT8_MAX - INT8_MIN)  # 255 levels
    offset = _round_half_even(-wmin / scale) + INT8_MIN  # zero point in [-128,...]
    return scale, offset


def quantize_weight_int8(
    w: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Quantize weights with the checkpoint order ``q = round(w/scale + offset)``.

    ``scale`` and ``offset`` are per-output-channel ``[out_channels, 1]``.
    """
    w = w.float()
    q = _round_half_even(w / scale + offset).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return q


def dequantize_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Dequantize weights: ``w = (q - offset) * scale`` (offset subtracted first)."""
    return (q.to(torch.float32) - offset) * scale


def w8a8_dynamic_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_offset: torch.Tensor,
) -> torch.Tensor:
    """Reference W8A8 dynamic linear: ``y = dequant(quant(x)) @ dequant(w).T``.

    Args:
        x: ``[T, in]`` float activations.
        weight_int8: ``[out, in]`` int8 weights.
        weight_scale, weight_offset: ``[out, 1]`` per-channel params.

    Returns:
        ``[T, out]`` float output.
    """
    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a)
    w_deq = dequantize_weight(weight_int8, weight_scale, weight_offset)
    return x_deq @ w_deq.t()


__all__ = [
    "INT8_MIN",
    "INT8_MAX",
    "quantize_per_token_int8",
    "dequantize_per_token",
    "compute_weight_qparams",
    "quantize_weight_int8",
    "dequantize_weight",
    "w8a8_dynamic_linear",
]
