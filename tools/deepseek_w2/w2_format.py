# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed W2 (and W4) weight layout for DeepSeek V4.1 on Ascend 310P (plan E1.1).

This module is the *canonical* on-disk layout for the 2-bit routed-expert
weights (and the 4-bit Engram weights) of the 552B-at-2-bit deployment. It is
the format the streaming converter (``w2_convert.py``, E1.1) writes and that the
weight loader (E1.2) and the INT8 grouped-matmul kernel bridge (E1.3) read back.
It is bit-for-bit identical to the E0.4 pure-PyTorch reference
(``tests/ut/deepseek_w2/reference/w2_moe_reference.py``); the E1.1 test asserts
that equality so the contract can never silently drift.

Format spec
===========

Codes
-----
Weights are quantised to *signed two's-complement* integer codes on a symmetric
grid. For ``n`` bits the code range is ``[-2**(n-1), 2**(n-1) - 1]``:

* **W2** (routed experts): 2-bit codes ``{-2, -1, 0, 1}``.
* **W4** (Engram): 4-bit codes ``{-8, ..., 7}``.

The grid is intentionally the honest on-device asymmetric two's-complement
range (not a clamped symmetric ``{-1, 0, 1}``): it wastes no code point, and the
per-block scale is chosen from *both* tails (see below) so every weight still
lands within half a quantisation step of a code.

Per-block scale
---------------
One float scale per ``[BLOCK_ROWS, BLOCK_COLS] = [32, 32]`` block of the
``[out_features, in_features]`` weight. ``[32, 32]`` is chosen to align with:

* the source ``weight_block_size = [32, 32]`` of the FP8/FP4 checkpoint, and
* the downstream INT8 grouped-matmul ``w13`` / ``w2`` per-output-channel layout,
  so E1.3 consumes one scale grid without re-blocking.

The scale of a block is::

    scale = max(pos_max / (code_max + 0.5), neg_absmax / (-(code_min) + 0.5), SCALE_FLOOR)

i.e. it is sized from the larger of the two tails so that ``w / scale`` for every
element of the block falls inside ``[code_min - 0.5, code_max + 0.5]`` and
therefore round-to-nearest never clips beyond half a step. The reconstruction
error is bounded by ``scale / 2`` per element.

Packing / endianness
--------------------
Codes are packed along the **last (input-feature) axis**, ``CODES_PER_BYTE =
8 // n_bits`` codes per ``uint8`` (W2: 4/byte, W4: 2/byte). Within a byte the
layout is **little-endian by field index**: code ``j`` of the group occupies
bits ``n*j .. n*j + n - 1``. Concretely, for W2 the first code along the input
axis is bits ``0..1`` (the low field) and the last is bits ``6..7``; for W4 the
first code is the low nibble and the second the high nibble. Only the low
``n`` bits of each (two's-complement) code are stored; unpacking sign-extends.

The packed tensor has shape ``[out_features, in_features // CODES_PER_BYTE]``
(``uint8``); the block-scale grid has shape
``[out_features // 32, in_features // 32]``.
"""

from __future__ import annotations

import torch

# --- block geometry (source weight_block_size + downstream w13/w2 grid) ------
W2_BLOCK_ROWS = 32
W2_BLOCK_COLS = 32

# --- signed two's-complement grids -------------------------------------------
# W2 routed experts: 2-bit codes {-2, -1, 0, 1}.
W2_BITS = 2
W2_CODE_MIN = -2
W2_CODE_MAX = 1
W2_CODES_PER_BYTE = 4  # 8 // W2_BITS

# W4 Engram: 4-bit codes {-8, ..., 7}.
W4_BITS = 4
W4_CODE_MIN = -8
W4_CODE_MAX = 7
W4_CODES_PER_BYTE = 2  # 8 // W4_BITS

# Smallest representable scale; guards all-zero / tiny blocks against 0-division.
SCALE_FLOOR = 1e-12

# Half-step nearest-rounding cover half-widths for the W2 asymmetric grid.
_W2_POS_COVER = W2_CODE_MAX + 0.5  # 1.5
_W2_NEG_COVER = -(W2_CODE_MIN) + 0.5  # 2.5


def _grid(n_bits: int) -> tuple[int, int, int, float, float]:
    """Return ``(code_min, code_max, codes_per_byte, pos_cover, neg_cover)``."""
    code_min = -(1 << (n_bits - 1))
    code_max = (1 << (n_bits - 1)) - 1
    codes_per_byte = 8 // n_bits
    return code_min, code_max, codes_per_byte, code_max + 0.5, -code_min + 0.5


def compute_block_scales(
    w: torch.Tensor,
    n_bits: int = W2_BITS,
    block_rows: int = W2_BLOCK_ROWS,
    block_cols: int = W2_BLOCK_COLS,
    method: str = "minmax",
    mse_num_candidates: int = 48,
) -> torch.Tensor:
    """Per-block scales for a weight matrix, one scalar per ``[32, 32]`` block.

    Args:
        w: ``[out, in]`` float weights; both dims multiples of the block size.
        n_bits: code width (2 for W2, 4 for W4).
        block_rows, block_cols: block shape.
        method: ``"minmax"`` (default, the original scale = ``max(pos_max/pos_cover,
            neg_absmax/neg_cover)`` -- sizes the scale so no value clips) or
            ``"mse"`` (per-block scale that MINIMISES the round-trip reconstruction
            MSE of the actual ``round -> clamp -> dequant`` grid). ``minmax`` wastes
            code resolution on rare block outliers; for W2 (only 4 levels) it caps
            weight cosine at ~0.83 vs the fp8 reference, while ``mse`` (optimum near
            ~0.30*absmax for W2) reaches ~0.92 at zero extra memory -- same code
            format, same kernels, just a better scale value. Measured on
            GLM-5.3-Flash experts against the fp8 source. ``mse`` also helps W4
            slightly (the ceiling is already ~0.99 there).
        mse_num_candidates: number of scale candidates searched per block (``mse``).

    Returns:
        ``[out // block_rows, in // block_cols]`` float64 per-block scales.
    """
    w = w.double()
    out_features, in_features = w.shape
    if out_features % block_rows or in_features % block_cols:
        raise ValueError(f"weight shape {tuple(w.shape)} must tile [{block_rows}, {block_cols}] blocks exactly.")
    code_min, code_max, _, pos_cover, neg_cover = _grid(n_bits)
    blocks = w.view(
        out_features // block_rows,
        block_rows,
        in_features // block_cols,
        block_cols,
    ).permute(0, 2, 1, 3)  # [Br, Bc, block_rows, block_cols]
    if method == "minmax":
        pos_max = blocks.clamp_min(0).amax(dim=(-2, -1))
        neg_absmax = blocks.clamp_max(0).abs().amax(dim=(-2, -1))
        scale = torch.maximum(pos_max / pos_cover, neg_absmax / neg_cover)
        return scale.clamp_min(SCALE_FLOOR)
    if method != "mse":
        raise ValueError(f"unknown scale method {method!r}; expected 'minmax' or 'mse'")

    # MSE-optimal per-block scale: search scale = frac * absmax over a grid and
    # keep, per block, the frac that minimises the reconstruction error of the
    # real quantiser (round(w/scale).clamp(code_min, code_max) * scale). Fully
    # vectorised over blocks; one pass per candidate frac.
    n_br, n_bc = out_features // block_rows, in_features // block_cols
    bflat = blocks.reshape(n_br, n_bc, block_rows * block_cols)  # [Br, Bc, B]
    absmax = bflat.abs().amax(dim=-1)  # [Br, Bc]
    # The minmax scale is the largest sensible candidate (frac ~ 1/pos_cover);
    # the MSE optimum for coarse grids sits well below it, so search (0, ~1].
    fracs = torch.linspace(1.0 / mse_num_candidates, 1.0, mse_num_candidates, dtype=w.dtype)
    best_err = None
    best_scale = None
    for frac in fracs.tolist():
        s = (absmax * frac).clamp_min(SCALE_FLOOR)  # [Br, Bc]
        s_e = s.unsqueeze(-1)
        codes = torch.round(bflat / s_e).clamp(code_min, code_max)
        err = ((codes * s_e - bflat) ** 2).sum(dim=-1)  # [Br, Bc]
        if best_err is None:
            best_err, best_scale = err, s
        else:
            better = err < best_err
            best_scale = torch.where(better, s, best_scale)
            best_err = torch.where(better, err, best_err)
    return best_scale.clamp_min(SCALE_FLOOR)


def broadcast_block_scales(
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    block_rows: int = W2_BLOCK_ROWS,
    block_cols: int = W2_BLOCK_COLS,
) -> torch.Tensor:
    """Expand ``[Br, Bc]`` block scales to the full ``[out, in]`` grid."""
    return (
        block_scale.double()
        .repeat_interleave(block_rows, dim=0)
        .repeat_interleave(block_cols, dim=1)[:out_features, :in_features]
    )


def quantize_weight(
    w: torch.Tensor,
    n_bits: int = W2_BITS,
    block_rows: int = W2_BLOCK_ROWS,
    block_cols: int = W2_BLOCK_COLS,
    method: str = "minmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise float weights to signed int codes + per-block scales.

    ``method`` selects the block-scale rule (see :func:`compute_block_scales`):
    ``"minmax"`` (default, no-clip) or ``"mse"`` (reconstruction-MSE-optimal,
    strongly recommended for W2 -- lifts weight cosine ~0.83 -> ~0.92 at no memory
    cost). The output format is identical either way, so ``mse``-quantised weights
    are a drop-in for the same dequant path and kernels.

    Returns ``(codes, block_scale)`` with ``codes`` int8 in
    ``[code_min, code_max]`` shaped ``[out, in]`` and ``block_scale`` float64
    ``[out // block_rows, in // block_cols]``.
    """
    w = w.double()
    out_features, in_features = w.shape
    code_min, code_max, _, _, _ = _grid(n_bits)
    block_scale = compute_block_scales(w, n_bits, block_rows, block_cols, method=method)
    full_scale = broadcast_block_scales(block_scale, out_features, in_features, block_rows, block_cols)
    codes = torch.round(w / full_scale).clamp(code_min, code_max).to(torch.int8)
    return codes, block_scale


def pack_codes(codes: torch.Tensor, n_bits: int = W2_BITS) -> torch.Tensor:
    """Pack signed int codes (last dim) into ``uint8``, little-endian by field.

    Args:
        codes: ``[..., in]`` int8 codes; ``in`` a multiple of ``8 // n_bits``.

    Returns:
        ``[..., in // codes_per_byte]`` uint8.
    """
    codes_per_byte = 8 // n_bits
    field_mask = (1 << n_bits) - 1
    if codes.shape[-1] % codes_per_byte:
        raise ValueError(f"last dim must be a multiple of {codes_per_byte} to pack {n_bits}-bit codes")
    fields = (codes.to(torch.int64) & field_mask).to(torch.uint8)
    groups = fields.view(*fields.shape[:-1], fields.shape[-1] // codes_per_byte, codes_per_byte)
    packed = torch.zeros(groups.shape[:-1], dtype=torch.uint8)
    for j in range(codes_per_byte):
        packed = packed | (groups[..., j] << (n_bits * j))
    return packed


def unpack_codes(packed: torch.Tensor, in_features: int, n_bits: int = W2_BITS) -> torch.Tensor:
    """Inverse of :func:`pack_codes`; returns sign-extended int8 codes.

    Args:
        packed: ``[..., in // codes_per_byte]`` uint8.
        in_features: original last-dim length.

    Returns:
        ``[..., in]`` int8 codes in ``[code_min, code_max]``.
    """
    codes_per_byte = 8 // n_bits
    field_mask = (1 << n_bits) - 1
    sign_wrap = 1 << n_bits
    half_wrap = 1 << (n_bits - 1)
    if packed.shape[-1] * codes_per_byte != in_features:
        raise ValueError("in_features does not match packed width")
    packed = packed.to(torch.int64)
    fields = []
    for j in range(codes_per_byte):
        field = (packed >> (n_bits * j)) & field_mask
        # Sign-extend: fields with the top bit set are negative.
        field = torch.where(field >= half_wrap, field - sign_wrap, field)
        fields.append(field)
    codes = torch.stack(fields, dim=-1).reshape(*packed.shape[:-1], in_features)
    return codes.to(torch.int8)


def dequantize_packed(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    n_bits: int = W2_BITS,
    block_rows: int = W2_BLOCK_ROWS,
    block_cols: int = W2_BLOCK_COLS,
) -> torch.Tensor:
    """Unpack packed codes and apply the per-block scale → float64 weights.

    Returns ``[out, in]`` float64 dequantised weights (codes × block scale).
    """
    codes = unpack_codes(packed, in_features, n_bits).double()
    full_scale = broadcast_block_scales(block_scale, out_features, in_features, block_rows, block_cols)
    return codes * full_scale


# --- W2/W4 named convenience wrappers (contract names for E1.2 / E1.3) -------
def quantize_weight_w2(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_weight(w, W2_BITS)


def pack_w2_codes(codes: torch.Tensor) -> torch.Tensor:
    return pack_codes(codes, W2_BITS)


def unpack_w2_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    return unpack_codes(packed, in_features, W2_BITS)


def dequantize_w2(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return dequantize_packed(packed, block_scale, out_features, in_features, W2_BITS)


def quantize_weight_w4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_weight(w, W4_BITS)


def pack_w4_codes(codes: torch.Tensor) -> torch.Tensor:
    return pack_codes(codes, W4_BITS)


def unpack_w4_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    return unpack_codes(packed, in_features, W4_BITS)


def dequantize_w4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return dequantize_packed(packed, block_scale, out_features, in_features, W4_BITS)


__all__ = [
    "W2_BLOCK_ROWS",
    "W2_BLOCK_COLS",
    "W2_BITS",
    "W2_CODE_MIN",
    "W2_CODE_MAX",
    "W2_CODES_PER_BYTE",
    "W4_BITS",
    "W4_CODE_MIN",
    "W4_CODE_MAX",
    "W4_CODES_PER_BYTE",
    "SCALE_FLOOR",
    "compute_block_scales",
    "broadcast_block_scales",
    "quantize_weight",
    "pack_codes",
    "unpack_codes",
    "dequantize_packed",
    "quantize_weight_w2",
    "pack_w2_codes",
    "unpack_w2_codes",
    "dequantize_w2",
    "quantize_weight_w4",
    "pack_w4_codes",
    "unpack_w4_codes",
    "dequantize_w4",
]
