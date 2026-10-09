# SPDX-License-Identifier: Apache-2.0
"""Explicit W4 grouped prefill resource; unchanged Cube products and rounding."""

import torch

from tools.qwen4exp.native_prefill import NativeResource, _tensors_on_one_device


class NativeCachedMetadata(NativeResource):
    def __call__(self, bank, prepared, group_ends):
        if len(prepared) != 4:
            raise ValueError("four native activation operands required")
        low, high, scales, sums = prepared
        weights = (bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum)
        if low.ndim != 2:
            raise ValueError("native packed activation must be two-dimensional")
        rows, packed_k = low.shape
        experts, outputs, weight_k = bank.weight.shape
        groups = packed_k // 64
        if (
            not 128 < rows <= 25600
            or experts != 128
            or outputs % 64
            or not 64 <= packed_k <= 1280
            or packed_k % 64
            or weight_k != packed_k
            or high.shape != low.shape
            or low.dtype != torch.int8
            or high.dtype != torch.int8
            or bank.weight.dtype != torch.int8
            or any(t.shape != (rows, groups, 8) or t.dtype != torch.float32 for t in (scales, sums))
            or any(t.shape != (experts, outputs, groups) or t.dtype != torch.float16 for t in weights[1:])
            or group_ends.shape != (experts,)
            or group_ends.dtype != torch.int64
        ):
            raise ValueError("unsupported metadata-cache prefill geometry/dtype")
        _tensors_on_one_device((*prepared, *weights, group_ends), low.device)
        output = torch.empty((rows, outputs), dtype=torch.float16, device=low.device)
        config = self.config((rows, experts, outputs, packed_k * 2, 8, 0, 1), low.device)
        self.launch(self.kernel, [*prepared, *weights, group_ends, output, config], 8)
        return output
