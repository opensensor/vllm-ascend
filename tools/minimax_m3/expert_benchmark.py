# SPDX-License-Identifier: Apache-2.0
"""Compare scale storage, lossless packing and batching with real expert weights."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from tools.glm_perf.operator_bench import nzpacked_codes
from tools.minimax_m3.layer_probe import metrics
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts, pack_w4_nz256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--iterations", type=int, default=10)
    packing = parser.add_mutually_exclusive_group()
    packing.add_argument("--nz", action="store_true", help="Test current-kernel 16x256 resident packing")
    packing.add_argument("--nz128", action="store_true", help="Test the older installed 16x128 NZ kernel layout")
    parser.add_argument("--route-cap", type=int, default=128)
    args = parser.parse_args()
    if args.iterations < 1 or args.route_cap not in (128, 512):
        parser.error("iterations must be positive and route-cap must be 128 or 512")
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(args.binding))
    weights = load_file(args.bundle / "weights.safetensors")
    with torch.device(args.device):
        bank = PackedM3Experts(6144, 3072, 128, 2, 0, 4, prefix="layers.3.experts")
    for name, tensor in bank.named_buffers():
        tensor.copy_(weights[name][None].expand_as(tensor))
    nz_codes = {}
    if args.nz:
        for projection in ("w1", "w2", "w3"):
            name = projection + "_codes"
            canonical = weights[name]
            nz_codes[name] = pack_w4_nz256(canonical)
    if args.nz128:
        for projection in ("w1", "w2", "w3"):
            name = projection + "_codes"
            canonical = weights[name]
            nz_codes[name] = nzpacked_codes(canonical[None], canonical.shape[1] * 2, 4)[0].view(torch.int8)
    reports = []
    with torch.inference_mode():
        for tokens in (1, 32, 128):
            x = torch.randn(tokens, 6144, generator=torch.Generator().manual_seed(79)).half().to(args.device)
            logits = torch.randn(tokens, 128, generator=torch.Generator().manual_seed(83)).to(args.device)
            variants = {}
            baseline = None
            configurations = [("canonical-fp16", torch.float16), ("canonical-fp32", torch.float32)]
            if args.route_cap != 128:
                configurations.append(("canonical-large-batch", torch.float16))
            if args.nz:
                configurations.append(("nz256-fp16", torch.float16))
            if args.nz128:
                configurations.append(("nz128-fp16", torch.float16))
            for variant, dtype in configurations:
                bank.max_grouped_routes = args.route_cap if variant not in ("canonical-fp16", "canonical-fp32") else 128
                for projection in ("w1", "w2", "w3"):
                    name = projection + "_scales"
                    setattr(bank, name, getattr(bank, name).to(dtype))
                    name = projection + "_codes"
                    source = nz_codes[name] if variant.startswith("nz") else weights[name]
                    if getattr(bank, name).dtype != source.dtype:
                        setattr(bank, name, torch.empty_like(getattr(bank, name), dtype=source.dtype))
                    getattr(bank, name).copy_(source[None].expand_as(getattr(bank, name)))
                for _ in range(2):
                    bank(x, logits)
                torch.npu.synchronize()
                durations = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    result = bank(x, logits)
                    torch.npu.synchronize()
                    durations.append(time.perf_counter() - start)
                if baseline is None:
                    baseline = result.cpu()
                variants[variant] = {
                    "median_seconds": statistics.median(durations),
                    "durations_seconds": durations,
                    "comparison": metrics(result, baseline),
                    "route_cap": bank.max_grouped_routes,
                    "resident_scale_bytes": sum(getattr(bank, p + "_scales").nbytes for p in ("w1", "w2", "w3")),
                }
            reports.append({"tokens": tokens, "variants": variants})
            print(json.dumps(reports[-1]), flush=True)
    args.output.write_text(
        json.dumps(
            {"scope": "One real expert repeated across 64 local slots; no TP collectives", "cases": reports}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
