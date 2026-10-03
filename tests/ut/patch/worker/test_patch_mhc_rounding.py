# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.patch.worker.patch_mhc_norm import _round_mhc_state


def test_mhc_default_rounding_matches_reference_bits():
    bits = torch.tensor(
        [0x3F808000, 0x3F818000, -0x407F8000, -0x407E8000, 0, -0x80000000],
        dtype=torch.int32,
    )
    value = bits.view(torch.float32)
    expected = ((bits + 0x7FFF + ((bits >> 16) & 1)) & -65536).view(torch.float32)

    actual = _round_mhc_state(value)

    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)


def test_mhc_default_rounding_matches_reference_for_finite_patterns():
    generator = torch.Generator().manual_seed(20261003)
    bits = torch.randint(0, 2**31, (10_000,), dtype=torch.int64, generator=generator).to(torch.int32)
    bits &= 0x7F7FFFFF  # Exclude NaN/Inf encodings; retain subnormals and large exponents.
    values = bits.view(torch.float32)
    values[1::2].neg_()
    expected_bits = values.view(torch.int32)
    expected = ((expected_bits + 0x7FFF + ((expected_bits >> 16) & 1)) & -65536).view(torch.float32)

    actual = _round_mhc_state(values)

    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)


def test_mhc_fp16_rounding_matches_cast_round_trip():
    value = torch.tensor([-7.01, -1.0078125, 0.0, 1.0078125, 7.01], dtype=torch.float32)

    actual = _round_mhc_state(value, use_fp16=True)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, value.to(torch.float16).to(torch.float32), rtol=0, atol=0)
