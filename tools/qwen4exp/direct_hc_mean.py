# SPDX-License-Identifier: Apache-2.0
"""Direct four-stream FP32/FP16 gated reduction with qualified summation order."""

import torch

from tools.qwen4exp.direct_hc_residual import HC_COUNT, MAX_TOKENS, MAX_WIDTH, TILE_WIDTH
from tools.qwen4exp.resident_candidates.fused_hc_projection import _is_capturing


def _supports_device(device):
    return device.type == "npu"


class DirectHCMean:
    def __init__(self, binary_path, *, kernel=None, launch=None):
        self.kernel = (
            kernel
            if kernel is not None
            else torch.classes.qwen_native_hc_mean_v1.Kernel(binary_path, "qwen_hc_mean_v1")
        )
        self.launch = launch if launch is not None else torch.ops.qwen_native_hc_mean_v1.launch
        self._tilings = {}

    def __call__(self, normalized, gate):
        if normalized.ndim != 2 or gate.shape != normalized.shape:
            raise ValueError("HC mean requires matching rank-two normalized values and gates")
        tokens, hyper_width = normalized.shape
        width = hyper_width // HC_COUNT
        if not 0 < tokens <= MAX_TOKENS or hyper_width % HC_COUNT or not 0 < width <= MAX_WIDTH or width % TILE_WIDTH:
            raise ValueError("HC mean requires four streams and supported width divisible by 512")
        if normalized.dtype != torch.float32 or gate.dtype != torch.float16:
            raise ValueError("HC mean requires FP32 normalized values and FP16 gates")
        if not _supports_device(normalized.device) or gate.device != normalized.device:
            raise ValueError("HC mean tensors must reside on one NPU")
        if not normalized.is_contiguous() or not gate.is_contiguous():
            raise ValueError("HC mean requires contiguous inputs")
        key = (tokens, width, normalized.device)
        tiling = self._tilings.get(key)
        if tiling is None:
            if _is_capturing(normalized.device):
                raise RuntimeError("warm up each HC mean shape before graph capture")
            tiling = torch.tensor([tokens, width], dtype=torch.int64, device=normalized.device)
            self._tilings[key] = tiling
        output = torch.empty((tokens, width), dtype=torch.float16, device=normalized.device)
        self.launch(self.kernel, [normalized, gate, output, tiling], 8)
        return output
