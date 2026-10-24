# SPDX-License-Identifier: Apache-2.0
"""Owned direct CANN binary dispatch for compatible NZ256 W4 candidates."""

import torch

from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256, reference_grouped_mm

CUBE_CORES = 8
OUTPUT_TILE = 128
MAX_ROUTES = 512


class GroupedW4Kernel:
    """Retain binary handles and shape scratch until the worker exits."""

    def __init__(self, namespace, binary, entry):
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, entry)
        self.launch = getattr(torch.ops, namespace).launch
        self.scratch = {}
        self.tilings = {}

    def __call__(self, x, codes, scales, ends):
        rows, cols = x.shape
        experts, output, packed_cols = codes.shape
        if (
            not 0 < rows <= MAX_ROUTES
            or cols < 256
            or cols % 256
            or output % OUTPUT_TILE
            or packed_cols * 2 != cols
            or experts <= 0
            or scales.shape != (experts, output // 32, cols // 32)
            or ends.shape != (experts,)
            or x.dtype != torch.float16
            or codes.dtype != torch.int8
            or scales.dtype != torch.float32
            or ends.dtype != torch.int64
        ):
            raise ValueError("native M3 candidate requires matching NZ256 W4 geometry and FP32 scales")
        blocks = min(CUBE_CORES, output // OUTPUT_TILE)
        key = (rows, experts, output, cols, x.device)
        if key not in self.tilings:
            self.tilings[key] = torch.tensor([rows, experts, output, cols, 2, 1], dtype=torch.int64, device=x.device)
        workspace_key = (blocks, cols, x.device)
        if workspace_key not in self.scratch:
            workspace = torch.empty(
                blocks * OUTPUT_TILE * cols * 2,
                dtype=torch.uint8,
                device=x.device,
            )
            self.scratch[workspace_key] = workspace
        tiling, workspace = self.tilings[key], self.scratch[workspace_key]
        result = torch.empty((rows, output), dtype=x.dtype, device=x.device)
        self.launch(self.kernel, [x, codes, scales, ends, result, workspace, tiling], blocks)
        return result


def validate_kernel(kernel):
    """Independent CPU W4 oracle, including concentrated routes and peer rows."""
    generator = torch.Generator().manual_seed(31)
    canonical = torch.randint(0, 256, (2, 256, 128), dtype=torch.uint8, generator=generator)
    codes = torch.stack([pack_w4_nz256(expert) for expert in canonical]).npu()
    scales = torch.rand((2, 8, 8), generator=generator) / 64
    device_scales = scales.npu()
    reports = []
    for rows, boundaries in ((4, [1, 3]), (128, [64, 128]), (512, [512, 512])):
        x = torch.randn(rows, 256, generator=generator).half()
        ends = torch.tensor(boundaries, dtype=torch.int64)
        expected = reference_grouped_mm(x, canonical, scales, ends)
        actual = kernel(x.npu(), codes, device_scales, ends.npu()).cpu()
        # Peer rows are deliberately unspecified by the native kernel; the
        # expert forward masks them before mixing results.
        actual = actual[: boundaries[-1]].float()
        expected = expected[: boundaries[-1]].float()
        error = float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt())
        reports.append({"rows": rows, "ends": boundaries, "nrmse": error, "finite": bool(actual.isfinite().all())})
    return {"passed": all(r["finite"] and r["nrmse"] < 0.001 for r in reports), "cases": reports}
