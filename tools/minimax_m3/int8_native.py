# SPDX-License-Identifier: Apache-2.0
"""Packed W4 / INT8 compute with immutable-bank scale metadata."""

import importlib

import torch

from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256, reference_grouped_mm

MAX_ROWS = 2048
MAX_COMPACT_ROUTES = 4


def prepare_scales(scales):
    # One scale per N32 block covers the signed code range [-8, 7] over all K.
    output_scale = (scales.float().amax(-1) * (8.0 / 127.0)).clamp_min(1e-12)
    return (scales.float() / output_scale[..., None]).contiguous(), output_scale.contiguous()


def pack_int8_layout(codes):
    """Lossless N16/K256 packed-byte reorder for INT8 B's K32/N16/K32 layout."""
    experts, rows, packed_cols = codes.shape
    return (
        codes.reshape(experts, rows // 16, packed_cols // 128, 4, 32, 16)
        .permute(0, 1, 2, 3, 5, 4)
        .contiguous()
        .reshape_as(codes)
    )


def paired_geometry(x, gate_codes, gate_scales, up_codes, up_scales, routes):
    """Check metadata only: route IDs stay on device, including peer sentinels."""
    if x.ndim != 2 or x.shape[0] != 1 or gate_codes.ndim != 3:
        raise ValueError("paired gate/up requires one input token and two matching expert banks")
    experts, output, packed_cols = gate_codes.shape
    cols = x.shape[1]
    if (
        not experts
        or not 1024 <= cols <= 6144
        or cols % 1024
        or not output
        or output % 128
        or packed_cols * 2 != cols
        or up_codes.shape != gate_codes.shape
        or gate_codes.dtype != torch.int8
        or up_codes.dtype != torch.int8
        or x.dtype != torch.float16
        or gate_scales.shape != (experts, output // 32, cols // 32)
        or up_scales.shape != gate_scales.shape
        or gate_scales.dtype not in (torch.float16, torch.float32)
        or up_scales.dtype not in (torch.float16, torch.float32)
        or routes.ndim != 1
        or not 1 <= routes.numel() <= MAX_COMPACT_ROUTES
        or routes.dtype != torch.int32
        or any(
            t.device != x.device or not t.is_contiguous()
            for t in (x, gate_codes, up_codes, gate_scales, up_scales, routes)
        )
    ):
        raise ValueError("paired gate/up requires FP16 single-token input, matching NZ W4 banks and <=4 INT32 routes")
    return routes.numel(), experts, output, cols


class GroupedW4Kernel:
    requires_int8_layout = True
    max_compact_routes = MAX_COMPACT_ROUTES

    def __init__(self, namespace, binary, entry, *, paired_binary=None, paired_entry=None):
        if (paired_binary is None) != (paired_entry is None):
            raise ValueError("paired binary and entry must be supplied together")
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, entry)
        self.launch = getattr(torch.ops, namespace).launch
        self.npu = importlib.import_module("torch_npu")
        self.scales = {}
        self.banks = {}
        ids = torch.arange(4096, dtype=torch.int64)
        offsets = (((ids // 512) * 32 + ids % 32) * 16 + (ids % 512) // 32) * 2
        self.offsets = offsets.to(torch.uint32).npu()
        self.tilings = {}
        if paired_binary is not None:
            self.paired_kernel = getattr(torch.classes, namespace).Kernel(paired_binary, paired_entry)
            self.paired_tilings = {}
            # Ordinary builds deliberately expose no paired callback.
            self.gate_up = self._gate_up

    def prepare(self, scales):
        version = None if torch.is_inference(scales) else scales._version
        key = (scales.data_ptr(), version, tuple(scales.shape), scales.device)
        if key not in self.scales:
            self.scales[key] = (scales, *prepare_scales(scales))
        return self.scales[key][1:]

    def bank_scales(self, codes, scales):
        # Codes identify immutable resident banks; transient FP32 scale casts
        # must not create one cache entry per generated token.
        key = (codes.data_ptr(), tuple(codes.shape), codes.device)
        if key not in self.banks:
            self.banks[key] = (codes, *prepare_scales(scales))
        return self.banks[key][1:]

    def __call__(self, x, codes, scales, ends):
        rows, cols = x.shape
        experts, output, packed_cols = codes.shape
        compact = ends.dtype == torch.int32 and ends.shape == (rows,) and rows <= MAX_COMPACT_ROUTES
        if (
            not 0 < rows <= MAX_ROWS
            or not 1024 <= cols <= 6144
            or cols % 1024
            or not output
            or output % 128
            or cols != packed_cols * 2
            or codes.dtype != torch.int8
            or x.dtype != torch.float16
            or scales.shape != (experts, output // 32, cols // 32)
            or not (compact or (ends.shape == (experts,) and ends.dtype == torch.int64))
        ):
            raise ValueError("INT8 kernel requires <=2048 rows, NZ W4, N128 and K1024 geometry")
        ratios, weight_scales = self.bank_scales(codes, scales)
        fused = rows <= MAX_COMPACT_ROUTES
        if fused:
            quantized, activation_scales = x, weight_scales
        else:
            quantized, activation_scales = self.npu.npu_dynamic_quant(x)
        key = (rows, experts, output, cols, x.device, compact)
        if key not in self.tilings:
            self.tilings[key] = torch.tensor(
                [rows, experts, output, cols, 2, 3 if compact else (2 if fused else 1)],
                dtype=torch.int64,
                device=x.device,
            )
        result = torch.empty((rows, output), dtype=x.dtype, device=x.device)
        self.launch(
            self.kernel,
            [
                quantized,
                codes,
                ratios,
                ends,
                result,
                result,
                self.tilings[key],
                activation_scales,
                weight_scales,
                self.offsets,
            ],
            min(8, output // 128),
        )
        return result

    def _gate_up(self, x, gate_codes, gate_scales, up_codes, up_scales, routes):
        rows, experts, output, cols = paired_geometry(x, gate_codes, gate_scales, up_codes, up_scales, routes)
        gate_ratios, gate_weights = self.bank_scales(gate_codes, gate_scales)
        up_ratios, up_weights = self.bank_scales(up_codes, up_scales)
        key = (rows, experts, output, cols, x.device)
        if key not in self.paired_tilings:
            self.paired_tilings[key] = torch.tensor(
                [rows, experts, output, cols, 2, 3], dtype=torch.int64, device=x.device
            )
        gate = torch.empty((rows, output), dtype=x.dtype, device=x.device)
        up = torch.empty_like(gate)
        self.launch(
            self.paired_kernel,
            [
                x,
                gate_codes,
                gate_ratios,
                routes,
                gate,
                gate,
                self.paired_tilings[key],
                gate_weights,
                gate_weights,
                self.offsets,
                up_codes,
                up_ratios,
                up_weights,
                up,
            ],
            min(8, output // 128),
        )
        return gate, up


def validate_paired(kernel, generator):
    reports = []
    # Distinct banks/scales, repeated routes, peer-first/all-peer routes, and
    # changing inputs cover projection identity and activation lifetime.
    for cols in (1024, 3072, 6144):
        banks, scales = [], []
        for _ in range(2):
            canonical = torch.randint(0, 256, (2, 256, cols // 2), dtype=torch.uint8, generator=generator)
            banks.append(pack_int8_layout(torch.stack([pack_w4_nz256(e) for e in canonical])).npu())
            scales.append((torch.rand((2, 8, cols // 32), generator=generator) / 64).npu())
        for ids in ([0], [1, 0, 1, 0], [2, 1, 0, 2], [2, 2, 2, 2]):
            x = torch.randn(1, cols, generator=generator).half().npu()
            routes = torch.tensor(ids, dtype=torch.int32, device=x.device)
            actual = kernel.gate_up(x, banks[0], scales[0], banks[1], scales[1], routes)
            expanded = x.expand(len(ids), -1).contiguous()
            expected = [kernel(expanded, bank, scale, routes) for bank, scale in zip(banks, scales)]
            valid = (routes < 2)[:, None]
            exact = [torch.equal(torch.where(valid, a, 0), torch.where(valid, b, 0)) for a, b in zip(actual, expected)]
            reports.append({"cols": cols, "ids": ids, "exact": all(exact)})
    return reports


def validate_kernel(kernel):
    generator = torch.Generator().manual_seed(31)
    canonical = torch.randint(0, 256, (2, 256, 512), dtype=torch.uint8, generator=generator)
    codes = torch.stack([pack_w4_nz256(expert) for expert in canonical])
    if kernel.requires_int8_layout:
        codes = pack_int8_layout(codes)
    codes = codes.npu()
    scales = torch.rand((2, 8, 32), generator=generator) / 64
    device_scales = scales.npu()
    reports = []
    for rows, boundaries in ((1, [1, 1]), (4, [1, 4]), (64, [32, 64]), (128, [128, 128]), (512, [512, 512])):
        x = torch.randn(rows, 1024, generator=generator).half()
        ends = torch.tensor(boundaries, dtype=torch.int64)
        expected = reference_grouped_mm(x, canonical, scales, ends)
        actual = kernel(x.npu(), codes, device_scales, ends.npu()).cpu()
        error = float(
            (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        )
        reports.append({"rows": rows, "nrmse": error, "finite": bool(actual.isfinite().all())})
    compact_reports = []
    for ids in ([0, 1, 2, 2], [2, 1, 0, 2], [2, 2, 2, 2], [1, 0, 1, 0]):
        x = torch.randn(len(ids), 1024, generator=generator).half().npu()
        route_ids = torch.tensor(ids, dtype=torch.int32, device=x.device)
        actual = kernel(x, codes, device_scales, route_ids)
        actual = torch.where((route_ids < codes.shape[0])[:, None], actual, 0).cpu()
        expected = torch.zeros_like(actual)
        for row, expert in enumerate(ids):
            if expert < codes.shape[0]:
                ends = torch.tensor([int(i >= expert) for i in range(codes.shape[0])], device=x.device)
                expected[row] = kernel(x[row : row + 1], codes, device_scales, ends).cpu()[0]
        compact_reports.append({"ids": ids, "exact": torch.equal(actual, expected)})
    paired_reports = validate_paired(kernel, generator) if callable(getattr(kernel, "gate_up", None)) else []
    return {
        "passed": (
            all(r["finite"] and r["nrmse"] < 0.025 for r in reports)
            and all(r["exact"] for r in compact_reports + paired_reports)
        ),
        "cases": reports,
        "compact_cases": compact_reports,
        "paired_cases": paired_reports,
    }
