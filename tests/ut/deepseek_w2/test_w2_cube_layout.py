# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only invariants for the 310P packed W2/W4 Cube layout.

These tests deliberately do not execute an NPU kernel.  They validate the bit
ordering, 16x16 NZ address mapping, and bounded-workspace contract used by
``w2_blocked_dequant_matmul_v310`` so the device run can be deferred safely.
"""

from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[3]
_KERNEL = _REPO_ROOT / "csrc/gmm/w2_blocked_dequant_matmul_v310/op_kernel/w2_blocked_dequant_matmul_v310.h"
_TILING = _REPO_ROOT / "csrc/gmm/w2_blocked_dequant_matmul_v310/op_host/w2_blocked_dequant_matmul_v310_tiling.cpp"
_BLOCK_MMAD = _REPO_ROOT / "csrc/moe/common/kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"

_BLOCK_K = 32
_FRACTAL = 16
_TILE_N = 128
_TILE_K = 128
_K_FRACTALS_PER_TILE = _TILE_K // _FRACTAL


def _pack_signed(codes: torch.Tensor, bits: int) -> torch.Tensor:
    codes_per_byte = 8 // bits
    mask = (1 << bits) - 1
    unsigned = codes.to(torch.int16) & mask
    fields = unsigned.reshape(*codes.shape[:-1], -1, codes_per_byte)
    packed = torch.zeros(fields.shape[:-1], dtype=torch.uint8)
    for field in range(codes_per_byte):
        packed |= (fields[..., field] << (bits * field)).to(torch.uint8)
    return packed


def _decode_like_kernel(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Re-express the kernel's field-major decode followed by its UB gather."""
    codes_per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign_half = 1 << (bits - 1)
    field_major = torch.cat([((packed.to(torch.int16) >> (bits * field)) & mask) for field in range(codes_per_byte)])
    signed = ((field_major + sign_half) & mask) - sign_half
    packed_cols = packed.numel()
    logical_to_field_major = torch.tensor(
        [field * packed_cols + byte for byte in range(packed_cols) for field in range(codes_per_byte)]
    )
    return signed[logical_to_field_major]


def _decode_tile_like_kernel(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Re-express the 16-row vector decode into [K-fractal,N,K] order."""
    codes_per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign_half = 1 << (bits - 1)
    rows, packed_cols = packed.shape
    flat = packed.flatten().to(torch.int16)
    field_major = torch.cat([((flat >> (bits * field)) & mask) for field in range(codes_per_byte)])
    signed = ((field_major + sign_half) & mask) - sign_half

    offsets = []
    for k_fractal in range(_K_FRACTALS_PER_TILE):
        for row in range(rows):
            for k_within in range(_FRACTAL):
                k_idx = k_fractal * _FRACTAL + k_within
                byte, field = divmod(k_idx, codes_per_byte)
                offsets.append(field * flat.numel() + row * packed_cols + byte)
    return signed[torch.tensor(offsets)].reshape(_K_FRACTALS_PER_TILE, rows, _FRACTAL)


def _to_zn(weight_t: torch.Tensor) -> torch.Tensor:
    """Pack logical B=[K,N] into CATLASS zN physical order."""
    k_dim, n_dim = weight_t.shape
    physical = torch.empty(k_dim * n_dim, dtype=weight_t.dtype)
    for k_idx in range(k_dim):
        for n_idx in range(n_dim):
            offset = (
                (k_idx // _FRACTAL) * (_FRACTAL * _FRACTAL)
                + (n_idx // _FRACTAL) * (k_dim * _FRACTAL)
                + (k_idx % _FRACTAL) * _FRACTAL
                + n_idx % _FRACTAL
            )
            physical[offset] = weight_t[k_idx, n_idx]
    return physical


def _from_zn(physical: torch.Tensor, k_dim: int, n_dim: int) -> torch.Tensor:
    logical = torch.empty((k_dim, n_dim), dtype=physical.dtype)
    for k_idx in range(k_dim):
        for n_idx in range(n_dim):
            offset = (
                (k_idx // _FRACTAL) * (_FRACTAL * _FRACTAL)
                + (n_idx // _FRACTAL) * (k_dim * _FRACTAL)
                + (k_idx % _FRACTAL) * _FRACTAL
                + n_idx % _FRACTAL
            )
            logical[k_idx, n_idx] = physical[offset]
    return logical


@pytest.mark.parametrize("bits", [2, 4])
def test_field_major_vector_decode_restores_logical_code_order(bits):
    low = -(1 << (bits - 1))
    high = (1 << (bits - 1)) - 1
    codes = torch.arange(_BLOCK_K, dtype=torch.int16) % (high - low + 1) + low
    packed = _pack_signed(codes, bits)

    torch.testing.assert_close(_decode_like_kernel(packed, bits), codes)


@pytest.mark.parametrize("bits", [2, 4])
def test_batched_tile_decode_emits_contiguous_16x16_fragments(bits):
    low = -(1 << (bits - 1))
    high = (1 << (bits - 1)) - 1
    codes = torch.arange(_FRACTAL * _TILE_K, dtype=torch.int16).reshape(_FRACTAL, _TILE_K)
    codes = codes % (high - low + 1) + low
    packed = _pack_signed(codes, bits)
    expected = codes.reshape(_FRACTAL, _K_FRACTALS_PER_TILE, _FRACTAL).permute(1, 0, 2)

    torch.testing.assert_close(_decode_tile_like_kernel(packed, bits), expected)


@pytest.mark.parametrize("bits", [2, 4])
def test_batched_decode_scale_and_transpose_matches_zn_weight(bits):
    low = -(1 << (bits - 1))
    high = (1 << (bits - 1)) - 1
    codes = torch.arange(_FRACTAL * _TILE_K, dtype=torch.int16).reshape(_FRACTAL, _TILE_K)
    codes = codes % (high - low + 1) + low
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float16)
    fragments = _decode_tile_like_kernel(_pack_signed(codes, bits), bits).to(torch.float16)
    got = torch.cat([(fragment.t() * scales[index // 2]).flatten() for index, fragment in enumerate(fragments)])

    scaled_weight = codes.to(torch.float16) * scales.repeat_interleave(_BLOCK_K)
    expected = _to_zn(scaled_weight.t().contiguous())
    torch.testing.assert_close(got, expected)


def test_transposed_weight_round_trips_through_zn_addressing():
    # Distinct values expose both a missing 16x16 transpose and swapped fractal
    # strides.  K=64 also covers both halves of more than one 32-column block.
    weight = torch.arange(_TILE_N * 64, dtype=torch.int32).reshape(_TILE_N, 64)
    physical = _to_zn(weight.t().contiguous())

    torch.testing.assert_close(_from_zn(physical, 64, _TILE_N), weight.t())


def test_kernel_and_tiling_keep_dequant_workspace_bounded_per_core():
    kernel = _KERNEL.read_text(encoding="utf-8")
    tiling = _TILING.read_text(encoding="utf-8")
    block_mmad = _BLOCK_MMAD.read_text(encoding="utf-8")

    assert "half, layout::zN" in kernel
    assert "DequantTileToNz" in kernel
    assert "for (uint32_t row = 0; row < W2_FRACTAL_SIZE; ++row)" in kernel
    assert "codesGm_[codeOffset + static_cast<int64_t>(row) * packedK_]" in kernel
    assert "W2_TILE_K = 128" in kernel
    assert "xfmGm_" not in kernel
    assert "N_ * K_ * sizeof(half)" not in kernel
    assert "static_cast<size_t>(blockDim) * static_cast<size_t>(OUTPUT_TILE)" in tiling
    assert "xfmBytes" not in tiling
    assert "310P unified-core output must match the accumulator type" in block_mmad
    assert "GlobalTensor<float> yfGm_" in kernel
    assert "CastOut(n0, nActual)" in kernel
    assert "rowCount = min(16U, mBlockActual - gmRow)" in block_mmad


def test_bounded_workspace_is_smaller_for_representative_expert():
    # Exclude the platform-owned system reserve, which is identical for both.
    tokens, n_dim, k_dim, cores = 48, 4096, 2048, 16
    m_aligned = (tokens + 15) // 16 * 16
    block_dim = min(n_dim // _TILE_N, cores)
    old_bytes = n_dim * k_dim * 2 + m_aligned * n_dim * 4 + cores * m_aligned * k_dim * 2
    new_bytes = block_dim * _TILE_N * k_dim * 2 + block_dim * m_aligned * _TILE_N * 4

    assert new_bytes < old_bytes
    assert new_bytes == 8_781_824
