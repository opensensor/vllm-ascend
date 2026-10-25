# SPDX-License-Identifier: Apache-2.0
"""Packed W4 / INT8 compute with immutable-bank scale metadata."""

import importlib

import torch

from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256, reference_grouped_mm

MAX_ROWS = 2048


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


class GroupedW4Kernel:
    requires_int8_layout = True

    def __init__(self, namespace, binary, entry):
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, entry)
        self.launch = getattr(torch.ops, namespace).launch
        self.npu = importlib.import_module("torch_npu")
        self.scales = {}
        self.banks = {}
        ids = torch.arange(4096, dtype=torch.int64)
        offsets = (((ids // 512) * 32 + ids % 32) * 16 + (ids % 512) // 32) * 2
        self.offsets = offsets.to(torch.uint32).npu()
        self.tilings = {}

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
            or ends.shape != (experts,)
            or ends.dtype != torch.int64
        ):
            raise ValueError("INT8 kernel requires <=2048 rows, NZ W4, N128 and K1024 geometry")
        ratios, weight_scales = self.bank_scales(codes, scales)
        fused = rows <= 4
        if fused:
            quantized, activation_scales = x, weight_scales
        else:
            quantized, activation_scales = self.npu.npu_dynamic_quant(x)
        key = (rows, experts, output, cols, x.device)
        if key not in self.tilings:
            self.tilings[key] = torch.tensor(
                [rows, experts, output, cols, 2, 2 if fused else 1], dtype=torch.int64, device=x.device
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
    return {"passed": all(r["finite"] and r["nrmse"] < 0.025 for r in reports), "cases": reports}
