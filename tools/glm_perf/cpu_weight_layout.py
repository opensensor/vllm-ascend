# SPDX-License-Identifier: Apache-2.0
"""Lossless CPU byte transcoding from canonical rows to native Cube tiles."""

import numpy as np
import torch

from .fused_weight_layout import CUBE_K, NZ_K, OUTPUT_TILE


def pack_canonical_cpu(packed, k):
    if type(k) is not int or k < NZ_K:
        raise ValueError("canonical input width must be a positive native tile multiple")
    if packed.device.type != "cpu" or packed.ndim != 2 or packed.dtype not in (torch.uint8, torch.int8):
        raise ValueError("expected CPU byte matrix of canonical codes")
    n, packed_k = packed.shape
    bits = packed_k * 8 // k
    if n < OUTPUT_TILE or n % OUTPUT_TILE or k % NZ_K or bits not in (2, 3, 4) or packed_k * 8 != k * bits:
        raise ValueError("canonical packing requires complete native tiles")
    raw = packed.contiguous().numpy().view(np.uint8)
    fields = 8 if bits == 3 else 8 // bits
    if bits == 3:
        raw = raw.reshape(n, k // fields, 3).astype(np.uint32)
        words = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16)
    else:
        words = raw
    unsigned = np.empty((n, k // fields, fields), dtype=np.uint8)
    for field in range(fields):
        unsigned[..., field] = (words >> (bits * field)) & ((1 << bits) - 1)
    unsigned = unsigned.reshape(n // OUTPUT_TILE, OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, CUBE_K)
    unsigned = unsigned.transpose(0, 2, 3, 1, 4)
    physical = np.concatenate((unsigned[..., ::2, :], unsigned[..., 1::2, :]), axis=-2)
    if bits == 4:
        pairs = physical.reshape(n // OUTPUT_TILE, k // NZ_K, -1, 2)
        encoded = pairs[..., 0] | (pairs[..., 1] << 4)
    else:
        planes = physical.reshape(n // OUTPUT_TILE, k // NZ_K, fields, -1)
        words = planes[..., 0, :].astype(np.uint32)
        for field in range(1, fields):
            words |= planes[..., field, :].astype(np.uint32) << (bits * field)
        encoded = np.stack([(words >> (8 * plane)) & 255 for plane in range(3 if bits == 3 else 1)], axis=-2)
    return torch.from_numpy(np.ascontiguousarray(encoded, dtype=np.uint8).reshape(n, packed_k))
