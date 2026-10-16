"""Opt-in vision normalization preserving the model's FP16 input boundary."""

import torch
from torch import nn

VISION_WIDTH = 1152


def eligible(x, zero, norm):
    return (
        x.ndim == 3
        and x.shape[1:] == (1, VISION_WIDTH)
        and 0 < x.shape[0] <= zero.shape[0]
        and x.dtype == zero.dtype == norm.weight.dtype == norm.bias.dtype == torch.float16
        and x.device == zero.device == norm.weight.device == norm.bias.device
        and x.is_contiguous()
    )


class VisionZeroAddNorm(nn.Module):
    def __init__(self, original, zero, normalize, verify=None):
        super().__init__()
        if (
            not isinstance(original, nn.LayerNorm)
            or original.normalized_shape != (VISION_WIDTH,)
            or original.weight is None
            or original.bias is None
            or zero.ndim != 3
            or zero.shape[1:] != (1, VISION_WIDTH)
            or zero.shape[0] <= 0
        ):
            raise ValueError("Use an affine 1152-wide LayerNorm and a bounded three-dimensional zero buffer")
        self.original = original
        self.register_buffer("zero", zero, persistent=False)
        self.normalize = normalize
        self.verify = verify

    def forward(self, x):
        if not eligible(x, self.zero, self.original):
            return self.original(x)
        # Keep the caller's FP16 rounding boundary. The full sum output must
        # be requested: the 310P path without it failed numerical validation.
        output, _, _, _ = self.normalize(
            x, self.zero[: x.shape[0]], self.original.weight, self.original.bias, self.original.eps, True
        )
        if self.verify is not None:
            if not self.verify(x, output):
                raise ValueError("Vision normalization failed its first model-activation check")
            self.verify = None
        return output
