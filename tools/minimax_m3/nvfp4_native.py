# SPDX-License-Identifier: Apache-2.0
"""Versioned source-NVFP4 / FP16 Cube projection, without shard reloads."""

import torch

from tools.minimax_m3.nvfp4 import LOOKUP_SIZE, pack_source, unpack_source

MAX_ROWS = 2048
MAX_COMPACT_ROUTES = 4


def geometry(x, codes, scales, ends, lookup):
    if x.ndim != 2 or codes.ndim != 3:
        raise ValueError("NVFP4 requires matrix input and a three-dimensional expert bank")
    rows, cols = x.shape
    experts, output, packed_cols = codes.shape
    compact = ends.dtype == torch.int32 and ends.shape == (rows,) and rows <= MAX_COMPACT_ROUTES
    if (
        not 0 < rows <= MAX_ROWS
        or not 1024 <= cols <= 6144
        or cols % 1024
        or not experts
        or not output
        or output % 128
        or packed_cols * 2 != cols
        or x.dtype != torch.float16
        or codes.dtype != torch.int8
        or scales.dtype != torch.uint8
        or scales.shape != (experts, output, cols // 16)
        or lookup.dtype != torch.float16
        or lookup.shape != (experts, LOOKUP_SIZE)
        or not (compact or (ends.dtype == torch.int64 and ends.shape == (experts,)))
        or any(t.device != x.device or not t.is_contiguous() for t in (x, codes, scales, ends, lookup))
    ):
        raise ValueError("NVFP4 requires FP16 input, NZ codes, raw K16 scales, FP16 tables and device routes")
    return rows, cols, experts, output, compact


class GroupedW4Kernel:
    """Own binary and per-shape workspaces; does not modify existing W4 banks."""

    max_compact_routes = MAX_COMPACT_ROUTES

    def __init__(self, namespace, binary, entry):
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, entry)
        self.launch = getattr(torch.ops, namespace).launch
        self.tilings = {}
        self.scratch = {}

    def __call__(self, x, codes, scales, ends, lookup):
        rows, cols, experts, output, compact = geometry(x, codes, scales, ends, lookup)
        blocks = min(8, output // 128)
        key = (rows, cols, experts, output, compact, x.device)
        if key not in self.tilings:
            self.tilings[key] = torch.tensor(
                [rows, experts, output, cols, 2, 3 if compact else 1], dtype=torch.int64, device=x.device
            )
        workspace_key = (blocks, cols, x.device)
        if workspace_key not in self.scratch:
            self.scratch[workspace_key] = torch.empty(blocks * 128 * cols * 2, dtype=torch.uint8, device=x.device)
        # The native entry writes peer zeros too; no uncapturable ZerosLike op.
        result = torch.empty((rows, output), dtype=x.dtype, device=x.device)
        self.launch(
            self.kernel,
            [x, codes, scales, ends, result, self.scratch[workspace_key], self.tilings[key], lookup],
            blocks,
        )
        return result


def validate_kernel(kernel):
    generator = torch.Generator().manual_seed(31)
    reports = []
    for cols in (1024, 3072, 6144):
        packed = torch.randint(0, 256, (2, 256, cols // 2), dtype=torch.uint8, generator=generator)
        scales = torch.randint(0, 127, (2, 256, cols // 16), dtype=torch.uint8, generator=generator)
        banks = [
            pack_source(packed[e], scales[e].view(torch.float8_e4m3fn), torch.tensor(1024.0 + 512 * e))
            for e in range(2)
        ]
        codes, metadata, lookup = [torch.stack(t).npu() for t in zip(*banks)]
        weights = [unpack_source(*bank).float() for bank in banks]
        for rows, boundaries in ((1, [1, 1]), (4, [1, 4]), (129, [128, 129]), (256, [256, 256])):
            x = torch.randn(rows, cols, generator=generator).half()
            expected = torch.zeros(rows, 256).half()
            start = 0
            for expert, stop in enumerate(boundaries):
                expected[start:stop] = (x[start:stop].float() @ weights[expert].T).half()
                start = stop
            actual = kernel(x.npu(), codes, metadata, torch.tensor(boundaries).npu(), lookup).cpu()
            error = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))
            reports.append({"cols": cols, "rows": rows, "nrmse": error, "finite": bool(actual.isfinite().all())})
        for ids in ([1, 0, 1, 0], [2, 1, 0, 2], [2, 2, 2, 2]):
            x = torch.randn(4, cols, generator=generator).half()
            expected = torch.zeros(4, 256).half()
            for row, expert in enumerate(ids):
                if expert < 2:
                    expected[row] = (x[row].float() @ weights[expert].T).half()
            actual = kernel(x.npu(), codes, metadata, torch.tensor(ids, dtype=torch.int32).npu(), lookup).cpu()
            error = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))
            reports.append({"cols": cols, "routes": ids, "nrmse": error, "finite": bool(actual.isfinite().all())})
    return {"passed": all(r["finite"] and r["nrmse"] < 0.001 for r in reports), "cases": reports}
