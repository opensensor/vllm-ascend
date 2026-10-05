# SPDX-License-Identifier: Apache-2.0
"""Fused small-row Hadamard/BF16-round/FP16 cast; instance-owned constants."""

import torch


class Rotation:
    def __init__(self, binary, namespace="glm_rotation_v1"):
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, "glm_kpool_rotation_v1")
        self.launch = getattr(torch.ops, namespace).launch
        ids = torch.arange(512, dtype=torch.int32)
        offsets, signs = [], []
        for stage in range(7):
            stride = 1 << stage
            left = ids & ~stride
            offsets.extend((left * 4, (left + stride) * 4))
            signs.append(torch.where(ids & stride == 0, 1.0, -1.0))
        expand = torch.stack((torch.zeros_like(ids), ids + 512), dim=-1).flatten() * 2
        self.offsets = torch.cat((torch.stack(offsets).flatten(), expand)).npu()
        self.signs = torch.stack(signs).npu()
        self.tiling = {
            (rows, dtype): torch.tensor([rows * 32 * 128, int(dtype == torch.bfloat16)], dtype=torch.int64).npu()
            for rows in range(1, 9) for dtype in (torch.float16, torch.bfloat16)
        }

    def __call__(self, query):
        if (
            query.dtype not in (torch.float16, torch.bfloat16)
            or query.ndim != 3
            or query.shape[1:] != (32, 128)
            or (query.shape[0], query.dtype) not in self.tiling
            or not query.is_contiguous()
        ):
            raise ValueError("rotation requires contiguous FP16/BF16 [1..8,32,128]")
        output = torch.empty_like(query, dtype=torch.float16)
        self.launch(self.kernel, [query, self.offsets, self.signs, output, self.tiling[query.shape[0], query.dtype]], 8)
        return output
