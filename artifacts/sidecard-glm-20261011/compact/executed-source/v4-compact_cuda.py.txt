# SPDX-License-Identifier: Apache-2.0
"""Lossless CUDA W2/W3 storage candidate, preserving the qualified W4 path.

This module is an isolated experiment; service admission and placement retain
their qualified defaults. It owns ordinary CudaOracle allocations and closes
them through the same lifecycle. No global device state or reset is added.
"""

from .device_backend import CUDA_COMPACT_PLANES, CUDA_NIBBLE_PLANES, CudaOracle, encode_cuda_weights

COMPACT_VALUES = 16
BLOCK_VALUES = 32
SUPPORTED_BITS = (2, 3, 4)


def encode_compact_weights(values, bits):
    """Sixteen codes in aligned word planes; W4 keeps its existing layout."""
    import numpy as np

    if (
        type(bits) is not int
        or bits not in SUPPORTED_BITS
        or values.ndim != 3
        or values.dtype != np.int8
        or min(values.shape) < 1
        or values.shape[-1] % BLOCK_VALUES
        or values.shape[-2] % BLOCK_VALUES
    ):
        raise ValueError("compact CUDA storage requires signed block32 weights and W2/W3/W4 precision")
    sign = 1 << (bits - 1)
    if values.min() < -sign or values.max() >= sign:
        raise ValueError("signed codes exceed their declared precision")
    if bits == 4:
        return encode_cuda_weights(values, CUDA_NIBBLE_PLANES)
    experts, channels, width = values.shape
    codes = (values.astype(np.int16) + sign).astype(np.uint64)
    codes = codes.reshape(experts, channels, width // BLOCK_VALUES, BLOCK_VALUES // COMPACT_VALUES, COMPACT_VALUES)
    codes = codes.transpose(0, 2, 3, 1, 4)
    packed = np.zeros(codes.shape[:-1], dtype=np.uint64)
    for lane in range(COMPACT_VALUES):
        packed |= codes[..., lane] << (bits * lane)
    if bits == 2:
        return packed.astype("<u4").tobytes()
    planes = np.stack([(packed >> (16 * word)).astype("<u2") for word in range(bits)], axis=-2)
    return planes.tobytes()


class CompactCudaOracle(CudaOracle):
    """Prototype adapter, only a bounded layer of CPU weights at a time."""

    def __init__(self, banks, specification, routes):
        spec = dict(specification, weight_layout=CUDA_COMPACT_PLANES, experimental_projection=False)
        super().__init__(banks, spec, routes)
