# SPDX-License-Identifier: Apache-2.0
"""Separate-process INT8/FP16 projection timing, not a W4 serving implementation."""

import argparse
import importlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from tools.minimax_m3.layer_probe import metrics
from vllm_ascend._310p.ops.minimax_m3 import unpack_w4

NZ_FORMAT = 29
MIN_FREE_BYTES = 1024**3
PROCESS_MEMORY_FRACTION = 0.02


def quantize_weight(weight):
    scale = weight.float().abs().amax(-1).clamp_min(1e-12) / 127
    packed = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return packed, scale


def timed(function, iterations):
    for _ in range(2):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = function()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - start)
    return result.cpu(), {"median_seconds": statistics.median(samples), "samples": samples}


def quantized_mm(npu, weight, scale, activation, row_scale):
    value = npu.npu_quant_matmul(activation, weight.t(), scale, output_dtype=torch.float16)
    return (value.float() * row_scale[:, None]).half()


def dynamic_mm(npu, weight, scale, activation):
    quantized, row_scale = npu.npu_dynamic_quant(activation)
    return quantized_mm(npu, weight, scale, quantized, row_scale)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    npu = importlib.import_module("torch_npu")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_per_process_memory_fraction(PROCESS_MEMORY_FRACTION, args.device)
    free, total = torch.npu.mem_get_info()
    if free < MIN_FREE_BYTES:
        raise RuntimeError("isolated INT8 probe needs at least 1 GiB free device memory")
    device = f"npu:{args.device}"
    weights = load_file(args.bundle / "weights.safetensors")
    reports = []
    for projection in ("w1", "w2"):
        weight = unpack_w4(weights[projection + "_codes"], weights[projection + "_scales"]).half()
        quantized, scale = quantize_weight(weight)
        fp16_nz = npu.npu_format_cast(weight.to(device), NZ_FORMAT)
        int8_nz = npu.npu_format_cast(quantized.to(device), NZ_FORMAT)
        encoded_scale = npu.npu_trans_quant_param(scale.to(device), None)
        for rows in (1, 64):
            x_cpu = torch.randn(rows, weight.shape[1], generator=torch.Generator().manual_seed(53)).half()
            x = x_cpu.to(device)
            expected = F.linear(x_cpu.float(), weight.float()).half()
            qa, activation_scale = npu.npu_dynamic_quant(x)

            variants = {}
            for name, function in (
                ("fp16_preexpanded", partial(F.linear, x, fp16_nz)),
                ("int8_prepared", partial(quantized_mm, npu, int8_nz, encoded_scale, qa, activation_scale)),
                ("int8_dynamic", partial(dynamic_mm, npu, int8_nz, encoded_scale, x)),
            ):
                actual, timing = timed(function, args.iterations)
                variants[name] = {**timing, "cpu_comparison": metrics(actual, expected)}
            record = {"projection": projection, "rows": rows, "shape": list(weight.shape), "variants": variants}
            reports.append(record)
            print(json.dumps(record), flush=True)
        del function, fp16_nz, int8_nz, encoded_scale, x, qa, activation_scale
        torch.npu.empty_cache()
    args.output.write_text(
        json.dumps(
            {
                "device": args.device,
                "initial_free_bytes": free,
                "total_bytes": total,
                "scope": (
                    "One real expert, preexpanded weights; excludes W4 unpack/requantization, "
                    "routing and collectives; not model tok/s"
                ),
                "cases": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
