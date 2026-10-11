# SPDX-License-Identifier: Apache-2.0
"""Weight-domain checks; hardware timing and model quality are separate gates."""

import pytest
import torch

from tools.qwen4exp.benchmark_int8_requant_projection_310 import requantize_weight


def test_zero_channels_retain_safe_unit_scale():
    dense, quantized, scale = requantize_weight(
        torch.zeros(160, 64, dtype=torch.int8),
        torch.full((160, 1), 0.125, dtype=torch.float16),
        torch.zeros(160, 1, dtype=torch.int8),
    )
    assert torch.count_nonzero(dense) == 0
    assert torch.count_nonzero(quantized) == 0
    assert torch.equal(scale, torch.ones(160))


def test_group_scales_are_folded_before_channel_quantization():
    # Signed 0x99 encodes -7 in both nibbles; centered codes are -6.
    packed = torch.full((160, 128), -103, dtype=torch.int8)
    scales = torch.tensor([0.125, 0.25], dtype=torch.float16).expand(160, -1).contiguous()
    offsets = torch.full((160, 2), -1, dtype=torch.int8)
    dense, quantized, channel_scale = requantize_weight(packed, scales, offsets)
    torch.testing.assert_close(dense[:, :128], torch.full((160, 128), -0.75), rtol=0, atol=0)
    torch.testing.assert_close(dense[:, 128:], torch.full((160, 128), -1.5), rtol=0, atol=0)
    assert quantized.dtype == torch.int8
    assert torch.equal(quantized[:, 128:], torch.full((160, 128), -127, dtype=torch.int8))
    torch.testing.assert_close(channel_scale, torch.full((160,), 1.5 / 127), rtol=0, atol=0)
    # This test explicitly checks the new lossy boundary rather than asserting
    # equivalence with the qualified G128 math.
    assert not torch.equal(quantized.float() * channel_scale.unsqueeze(1), dense)


def test_invalid_checkpoint_metadata_is_rejected():
    with pytest.raises(ValueError, match="metadata"):
        requantize_weight(
            torch.zeros(160, 64, dtype=torch.int8), torch.ones(160, 1), torch.zeros(160, 1, dtype=torch.int8)
        )
