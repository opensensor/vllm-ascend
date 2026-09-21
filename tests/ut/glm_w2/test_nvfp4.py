# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NVFP4 (Blackwell E2M1) decode primitives.

These pin the exact E2M1 codebook and the block-16 scale broadcast so the
310P eager dequant can never silently drift from the golden GPU NVFP4 weights.
"""

import torch

from tools.deepseek_w2.w2_format import (
    NVFP4_BLOCK_COLS,
    NVFP4_CODES_PER_BYTE,
    dequantize_nvfp4,
    unpack_nvfp4_codes,
)


def _ref_decode(packed: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """Reference E2M1 decode (the shipped ``kpool``/NVFP4 kernel semantics)."""
    out, in_bytes = packed.shape
    in_features = in_bytes * NVFP4_CODES_PER_BYTE
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=-1).reshape(out, in_features)
    mag = nibbles & 0x07
    sign = (nibbles >> 3) & 1
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    val = table[mag.to(torch.int64)]
    val = torch.where(sign == 1, -val, val)
    full = block_scale.double().repeat_interleave(NVFP4_BLOCK_COLS, dim=1)
    return val * full


def test_unpack_nvfp4_codes_e2m1_codebook():
    # Byte 0x21 -> nibbles [1, 2] -> [0.5, 1.0]; byte 0x6F -> [15, 6] -> [-6.0, 4.0].
    packed = torch.tensor([[0x21, 0x6F]], dtype=torch.uint8)
    out = unpack_nvfp4_codes(packed, 4)
    expected = torch.tensor([[0.5, 1.0, -6.0, 4.0]], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_dequantize_nvfp4_matches_reference():
    torch.manual_seed(0)
    out_features, in_features = 8, 32
    nibbles = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)  # [out, in//2]
    block_scale = torch.rand(out_features, in_features // NVFP4_BLOCK_COLS) * 0.01 + 0.01

    got = dequantize_nvfp4(packed, block_scale, out_features, in_features)
    ref = _ref_decode(packed, block_scale)
    assert got.shape == (out_features, in_features)
    assert torch.allclose(got, ref, atol=0.0, rtol=0.0)
