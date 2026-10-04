# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.patch.worker.patch_mhc_norm import MHC_AI_CORE_ROUND_MAX_ELEMENTS, _round_mhc_state


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


def test_mhc_golden_rounding_preserves_bf16_range():
    # The GLM golden keeps bf16 mHC state. A fp16 round trip clips a large
    # residual stream (to inf on CPU or max finite on 310P).
    value = torch.tensor([1.0, 65504.0, 70000.0], dtype=torch.float32)

    golden = _round_mhc_state(value)
    fp16 = _round_mhc_state(value, use_fp16=True)

    assert torch.isfinite(golden).all()
    assert fp16[-1] != golden[-1]
    torch.testing.assert_close(golden, value.to(torch.bfloat16).to(torch.float32), rtol=0, atol=0)


def test_mhc_ai_core_round_is_opt_in_and_bounded(monkeypatch):
    calls = []

    def fake_round(x):
        calls.append(x.numel())
        return x.to(torch.bfloat16).to(torch.float32)

    monkeypatch.setattr(torch.ops._C_ascend, "mhc_bf16_round_310", fake_round, raising=False)
    small = torch.randn(4116)
    expected = _round_mhc_state(small)
    torch.testing.assert_close(_round_mhc_state(small, use_ai_core=True), expected, rtol=0, atol=0)
    assert calls == [4116]

    boundary = torch.randn(MHC_AI_CORE_ROUND_MAX_ELEMENTS)
    _round_mhc_state(boundary, use_ai_core=True)
    assert calls == [4116, MHC_AI_CORE_ROUND_MAX_ELEMENTS]

    large = torch.randn(MHC_AI_CORE_ROUND_MAX_ELEMENTS + 1)
    _round_mhc_state(large, use_ai_core=True)
    _round_mhc_state(small[::2], use_ai_core=True)
    _round_mhc_state(small, use_fp16=True, use_ai_core=True)
    assert calls == [4116, MHC_AI_CORE_ROUND_MAX_ELEMENTS]
