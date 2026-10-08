"""Exhaustive CPU parity for candidate W2/W4 packed-field arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from tools.glm_perf.packed_fields import quotient_via_fp16_rint, unpack_signed_fields_fp16


@pytest.mark.parametrize("divisor", [4, 16, 64])
def test_fp16_biased_rint_matches_every_byte_quotient(divisor: int) -> None:
    packed = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(quotient_via_fp16_rint(packed, divisor), packed.astype(np.int16) // divisor)


@pytest.mark.parametrize("bits_per_code", [2, 4])
def test_shift_free_signed_fields_match_every_packed_byte(bits_per_code: int) -> None:
    packed = np.arange(256, dtype=np.uint8)
    divisor = 1 << bits_per_code
    mask = divisor - 1
    expected = []
    for field in range(8 // bits_per_code):
        unsigned = (packed.astype(np.int16) >> (field * bits_per_code)) & mask
        expected.append(np.where(unsigned >= divisor // 2, unsigned - divisor, unsigned))
    np.testing.assert_array_equal(unpack_signed_fields_fp16(packed, bits_per_code), np.stack(expected, axis=-1))


def test_candidate_rejects_unproven_types_and_divisors() -> None:
    packed = np.array([0, 255], dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported divisor"):
        quotient_via_fp16_rint(packed, 8)
    with pytest.raises(ValueError, match="unsupported code width"):
        unpack_signed_fields_fp16(packed, 1)
    with pytest.raises(TypeError, match="uint8"):
        unpack_signed_fields_fp16(packed.astype(np.int16), 4)
    with pytest.raises(TypeError, match="uint8"):
        quotient_via_fp16_rint(packed.astype(np.int16), 16)


def test_candidate_preserves_multidimensional_byte_shape() -> None:
    packed = np.array([[0x00, 0x7F], [0x80, 0xFF]], dtype=np.uint8)
    assert unpack_signed_fields_fp16(packed, 2).shape == (2, 2, 4)
    assert unpack_signed_fields_fp16(packed, 4).shape == (2, 2, 2)
