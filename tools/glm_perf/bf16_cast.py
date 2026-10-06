# SPDX-License-Identifier: Apache-2.0
"""Exact, graph-safe BF16 storage conversions through a qualified AI-Core kernel."""

from pathlib import Path

import torch


class NativeBF16Cast:
    def __init__(self, root, namespace):
        self.kernel = getattr(torch.classes, namespace).Kernel(
            str(Path(root) / "glm_bf16_cast.bin"), "glm_bf16_cast_v1"
        )
        self.launch = getattr(torch.ops, namespace).launch
        self.configs = {}

    def __call__(self, value, dtype):
        if value.dtype == dtype:
            return value
        if value.device.type != "npu" or torch.bfloat16 not in (value.dtype, dtype):
            return value.to(dtype)
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("native BF16 conversion supports FP16, FP32 and BF16")
        decode = value.dtype == torch.bfloat16
        mode = (3 if dtype == torch.float16 else 1) if decode else (2 if value.dtype == torch.float16 else 0)
        source = value.contiguous() if mode else value.float().contiguous()
        return self._convert(source, dtype, mode)

    def round(self, value, dtype=torch.float32):
        """Fuse FP32 -> BF16 -> FP32/FP16 without a BF16 GM temporary."""
        if value.dtype != torch.float32 or dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("BF16 rounding requires FP32 input and a floating output")
        if value.device.type != "npu":
            return value.bfloat16().to(dtype)
        mode = {torch.bfloat16: 0, torch.float32: 4, torch.float16: 5}[dtype]
        return self._convert(value.contiguous(), dtype, mode)

    def _convert(self, source, dtype, mode):
        result = torch.empty_like(source, dtype=dtype)
        if source.numel():
            key = (source.device, source.numel(), mode)
            if key not in self.configs:
                self.configs[key] = torch.tensor((source.numel(), mode), dtype=torch.int64, device=source.device)
            self.launch(self.kernel, [source, result, self.configs[key]], min(8, (source.numel() + 255) // 256))
        return result
