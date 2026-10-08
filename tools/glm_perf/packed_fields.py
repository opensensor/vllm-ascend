"""CPU proof for a shift-free W2/W4 byte extraction candidate.

This is not an NPU kernel. It checks the FP16 biased-RINT arithmetic used by
the existing Qwen W4 path as a possible replacement for GLM's per-field
mask/cast/reciprocal chain. Native operator cost and parity remain NPU gates.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_DIVISORS = (4, 16, 64)
SUPPORTED_CODE_BITS = (2, 4)


def quotient_via_fp16_rint(packed: np.ndarray, divisor: int) -> np.ndarray:
    """Compute floor(byte / divisor) without an integer shift.

    ``byte / divisor`` has fractional steps of ``1/divisor``. Subtracting
    ``(divisor-1)/(2*divisor)`` places every noninteger step strictly below
    the nearest-integer midpoint, and integer steps strictly above it. All
    values involved are dyadic and exactly representable in FP16 for bytes.
    """
    if divisor not in SUPPORTED_DIVISORS:
        raise ValueError(f"unsupported divisor {divisor}")
    bytes_u8 = np.asarray(packed)
    if bytes_u8.dtype != np.uint8:
        raise TypeError("packed codes must be uint8")
    value = bytes_u8.astype(np.float16)
    bias = np.float16(-(divisor - 1) / (2 * divisor))
    scaled = value * np.float16(1 / divisor)
    return np.rint(scaled + bias).astype(np.int16)


def unpack_signed_fields_fp16(packed: np.ndarray, bits_per_code: int) -> np.ndarray:
    """Extract low-to-high signed fields from every packed byte."""
    if bits_per_code not in SUPPORTED_CODE_BITS:
        raise ValueError(f"unsupported code width {bits_per_code}")
    bytes_u8 = np.asarray(packed)
    if bytes_u8.dtype != np.uint8:
        raise TypeError("packed codes must be uint8")
    divisor = 1 << bits_per_code
    quotients = [bytes_u8.astype(np.int16)]
    for field in range(1, 8 // bits_per_code):
        quotients.append(quotient_via_fp16_rint(bytes_u8, divisor**field))
    quotients.append(np.zeros_like(quotients[0]))
    signed_fields = []
    for field in range(8 // bits_per_code):
        unsigned = quotients[field] - divisor * quotients[field + 1]
        signed_fields.append(np.where(unsigned >= divisor // 2, unsigned - divisor, unsigned).astype(np.int16))
    return np.stack(signed_fields, axis=-1)
