# SPDX-License-Identifier: Apache-2.0
import torch

from tools.minimax_m3.int8_projection_probe import quantize_weight


def test_per_output_quantizer_handles_zero_channels_and_signed_extremes():
    weight = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [1.0, -3.0, 0.5]]).half()
    quantized, scale = quantize_weight(weight)
    assert quantized.dtype == torch.int8
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert quantized[0].count_nonzero() == 0
    assert quantized[1].tolist() == [-127, 0, 127]
    error = (quantized.float() * scale[:, None] - weight.float()).abs()
    assert (error <= scale[:, None] / 2 + 1e-6).all()
