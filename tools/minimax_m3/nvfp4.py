# SPDX-License-Identifier: Apache-2.0
"""Source-preserving NVFP4 tiles; no uniform INT4 requantization.

The lookup includes source FP32 block/global scaling followed by FP16 rounding.
It preserves decoded weights, not the source model's activation quantization.
"""

import torch

from tools.minimax_m3.mixed_checkpoint import E2M1_VALUES
from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256

GROUP = 16
TILE_N = 16
TILE_K = 256
LOOKUP_SIZE = 256 * 16
NVFP4_LAYOUT = "minimax-m31-nvfp4-nz256-scale16-v1"


def weight_lookup(global_scale):
    """One 8 KiB table per projection/expert; valid scale bytes decode exactly."""
    if (
        global_scale.device.type != "cpu"
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
        or not torch.isfinite(global_scale).all()
        or not (global_scale > 0).all()
    ):
        raise ValueError("NVFP4 requires one finite positive CPU FP32 global scale")
    scales = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    # Invalid scale encodings are rejected when packing actual weights.
    scales = torch.where(torch.isfinite(scales) & (scales >= 0), scales, 0)
    magnitudes = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    values = torch.cat((magnitudes, -magnitudes))
    return (values[None, :] * (scales / global_scale.reshape(()))[:, None]).half().flatten()


def pack_source(packed, scales, global_scale):
    """Return reversible NZ code/scale bytes and the exact FP16 decode table."""
    if (
        packed.device.type != "cpu"
        or packed.dtype != torch.uint8
        or packed.ndim != 2
        or not packed.shape[0]
        or packed.shape[0] % TILE_N
        or not packed.shape[1]
        or packed.shape[1] % (TILE_K // 2)
        or scales.device.type != "cpu"
        or scales.dtype != torch.float8_e4m3fn
        or scales.shape != (packed.shape[0], packed.shape[1] // (GROUP // 2))
    ):
        raise ValueError("Expected CPU NVFP4 codes/scales covering complete 16x256 tiles")
    values = scales.float()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    lookup = weight_lookup(global_scale)
    used = lookup.reshape(256, 16)[scales.view(torch.uint8).long()]
    if not torch.isfinite(used).all():
        raise ValueError("NVFP4 FP16 reconstruction overflow")
    rows, packed_cols = packed.shape
    tiled = (
        scales.view(torch.uint8)
        .reshape(rows // TILE_N, TILE_N, packed_cols * 2 // TILE_K, TILE_K // GROUP)
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape_as(scales)
    )
    return pack_w4_nz256(packed), tiled, lookup


def unpack_source(codes, scales, lookup):
    """CPU inverse layout and table oracle for a single expert projection."""
    if any(t.device.type != "cpu" for t in (codes, scales, lookup)):
        raise ValueError("NVFP4 oracle is CPU-only")
    rows, packed_cols = codes.shape
    tiles = codes.view(torch.uint8).reshape(rows // TILE_N, packed_cols * 2 // TILE_K, -1)
    fields = torch.cat((tiles & 15, tiles >> 4), -1).long()
    scale_tiles = scales.reshape(rows // TILE_N, packed_cols * 2 // TILE_K, TILE_K // GROUP, TILE_N)
    indices = fields + scale_tiles.repeat_interleave(GROUP, -2).flatten(-2).long() * 16
    decoded = lookup[indices].reshape(rows // TILE_N, packed_cols * 2 // TILE_K, TILE_K, TILE_N)
    return decoded.permute(0, 3, 1, 2).contiguous().reshape(rows, packed_cols * 2)


def storage_bytes(rows, cols, experts=1):
    if rows <= 0 or rows % TILE_N or cols <= 0 or cols % TILE_K or experts <= 0:
        raise ValueError("Expected positive, tile-aligned NVFP4 bank geometry")
    return experts * (rows * cols // 2 + rows * cols // GROUP + LOOKUP_SIZE * 2)
