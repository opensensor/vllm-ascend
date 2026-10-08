# SPDX-License-Identifier: Apache-2.0
"""Versioned direct AI Core HC residual kernel for trusted resident experiments."""

import torch

from tools.qwen4exp.resident_candidates.fused_hc_projection import _is_capturing

HC_COUNT = 4
TILE_WIDTH = 512
MAX_TOKENS = 8192
MAX_WIDTH = 16384


def _supports_device(device):
    return device.type == "npu"


class DirectHCResidual:
    def __init__(self, binary_path, *, kernel=None, launch=None):
        self.kernel = (
            kernel
            if kernel is not None
            else torch.classes.qwen_native_hc_residual_v1.Kernel(binary_path, "qwen_hc_residual_v1")
        )
        self.launch = launch if launch is not None else torch.ops.qwen_native_hc_residual_v1.launch
        self._tilings = {}

    def __call__(self, hyper, block, injection):
        if hyper.ndim != 2 or block.ndim != 2 or injection.ndim != 2:
            raise ValueError("HC residual expects rank-two tensors")
        tokens, width = block.shape
        if not 0 < tokens <= MAX_TOKENS or not 0 < width <= MAX_WIDTH or width % TILE_WIDTH:
            raise ValueError("HC residual requires a positive supported token count and width divisible by 512")
        if hyper.shape != (tokens, HC_COUNT * width) or injection.shape != (tokens, HC_COUNT):
            raise ValueError("HC residual expects four streams and matching token counts")
        if (
            hyper.dtype != torch.float16
            or block.dtype != torch.float16
            or injection.dtype
            not in (
                torch.float16,
                torch.float32,
            )
        ):
            raise ValueError("HC residual requires FP16 hyper/block and FP16 or FP32 injection")
        if not _supports_device(hyper.device) or any(t.device != hyper.device for t in (block, injection)):
            raise ValueError("HC residual tensors must reside on one NPU")
        if not all(t.is_contiguous() for t in (hyper, block, injection)):
            raise ValueError("HC residual requires contiguous inputs")
        key = (tokens, width, hyper.device)
        tiling = self._tilings.get(key)
        if tiling is None:
            if _is_capturing(hyper.device):
                raise RuntimeError("warm up each HC residual shape before graph capture")
            tiling = torch.tensor([tokens, width], dtype=torch.int64, device=hyper.device)
            self._tilings[key] = tiling
        output = torch.empty_like(hyper)
        coefficients = injection.float()
        self.launch(self.kernel, [hyper, block, coefficients, output, tiling], 8)
        return output
