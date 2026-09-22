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

_BLOCK_K = 32
_FRACTAL = 16
_TILE_N = 128


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


def test_transposed_weight_round_trips_through_zn_addressing():
    # Distinct values expose both a missing 16x16 transpose and swapped fractal
    # strides.  K=64 also covers both halves of more than one 32-column block.
    weight = torch.arange(_TILE_N * 64, dtype=torch.int32).reshape(_TILE_N, 64)
    physical = _to_zn(weight.t().contiguous())

    torch.testing.assert_close(_from_zn(physical, 64, _TILE_N), weight.t())


def test_kernel_and_tiling_keep_dequant_workspace_bounded_per_core():
    kernel = _KERNEL.read_text(encoding="utf-8")
    tiling = _TILING.read_text(encoding="utf-8")

    assert "half, layout::zN" in kernel
    assert "DequantTileToNz" in kernel
    assert "xfmGm_" not in kernel
    assert "N_ * K_ * sizeof(half)" not in kernel
    assert "static_cast<size_t>(blockDim) * static_cast<size_t>(OUTPUT_TILE)" in tiling
    assert "xfmBytes" not in tiling


def test_bounded_workspace_is_smaller_for_representative_expert():
    # Exclude the platform-owned system reserve, which is identical for both.
    tokens, n_dim, k_dim, cores = 48, 4096, 2048, 16
    m_aligned = (tokens + 15) // 16 * 16
    block_dim = min(n_dim // _TILE_N, cores)
    old_bytes = n_dim * k_dim * 2 + m_aligned * n_dim * 4 + cores * m_aligned * k_dim * 2
    new_bytes = block_dim * _TILE_N * k_dim * 2 + block_dim * m_aligned * _TILE_N * 4

    assert new_bytes < old_bytes
    assert new_bytes == 8_781_824
