# SPDX-License-Identifier: Apache-2.0
"""Versioned direct-kernel candidate; no OPP discovery or process-global cache."""

import torch


class DirectPrefillScore:
    def __init__(self, binary_path):
        self.kernel = torch.classes.glm_native_v1.Kernel(binary_path, "glm_kpool_prefill_score_v310")

    def __call__(self, query, weights, flat, table, ends, positions, pools, blocks, br, bs, rs, offset):
        if query.ndim != 3 or query.shape[1:] != (32, 128) or query.dtype != torch.float16:
            raise ValueError("direct prefill query must be FP16 [rows,32,128]")
        rows = query.shape[0]
        if (
            weights.shape != (rows, 32)
            or weights.dtype != torch.float32
            or flat.ndim != 1
            or flat.dtype != torch.float16
        ):
            raise ValueError("invalid direct prefill weights or cache")
        if (
            table.ndim != 2
            or table.dtype != torch.int32
            or positions.shape != (rows,)
            or positions.dtype != torch.int32
        ):
            raise ValueError("invalid direct prefill page table or positions")
        if ends.shape != (1,) or ends.dtype != torch.int32:
            raise ValueError("direct prefill needs one request end")
        if not 0 < rows <= 128 or rows % 4 or pools <= 0 or pools % 8 or table.shape[0] != 1:
            raise ValueError("invalid direct prefill geometry")
        if (
            pools > table.shape[1] * br
            or not 0 < br <= 2**31 - 1
            or bs < br * rs
            or bs % 16
            or rs < 128
            or rs % 16
            or rs // 16 > 65535
        ):
            raise ValueError("invalid direct prefill pages")
        if offset < 0 or offset % 16 or blocks <= 0 or (blocks - 1) * bs + (br - 1) * rs + 128 > flat.numel():
            raise ValueError("direct prefill cache view exceeds backing storage")
        tiling = torch.tensor(
            [rows, pools, 1, table.shape[1], blocks, br, bs, rs, offset], dtype=torch.int64, device=query.device
        )
        output = torch.full((rows, pools), -torch.inf, dtype=torch.float32, device=query.device)
        workspace = torch.empty(32, dtype=torch.uint8, device=query.device)
        torch.ops.glm_native_v1.launch(
            self.kernel, [query, weights, flat, table, ends, positions, output, workspace, tiling], 8
        )
        return output
