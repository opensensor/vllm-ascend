# SPDX-License-Identifier: Apache-2.0
"""Isolated W4A8 real-weight timing with conversion included; no serving mutation."""

import argparse
import importlib
import json
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file

from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3.int8_native import pack_int8_layout
from tools.minimax_m3.int8_projection_probe import MIN_FREE_BYTES, PROCESS_MEMORY_FRACTION, timed
from tools.minimax_m3.layer_probe import metrics
from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256, unpack_w4


def load_kernel(path):
    manifest = NativeManifest(json.loads(path.read_text()))
    manifest.verify_files()
    for item in manifest.value["libraries"]:
        torch.ops.load_library(item["path"])
    namespace = {}
    exec(manifest.value["validation_source"], namespace)
    return namespace["prepare"](), namespace["validate"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or (args.bundle is not None and args.baseline is None):
        parser.error("iterations must be positive; real-weight timing requires --baseline")
    importlib.import_module("torch_npu")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_per_process_memory_fraction(PROCESS_MEMORY_FRACTION, args.device)
    free, _ = torch.npu.mem_get_info()
    if free < MIN_FREE_BYTES:
        raise RuntimeError("isolated native probe requires at least 1 GiB free")
    device = f"npu:{args.device}"
    candidate, validate = load_kernel(args.candidate)
    validation = validate(candidate)
    report = {
        "validation": validation,
        "initial_free_bytes": free,
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "cases": [],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(validation), flush=True)
    if not validation["passed"]:
        raise RuntimeError("native validation failed")
    if args.bundle is None:
        return
    baseline, _ = load_kernel(args.baseline)
    weights = load_file(args.bundle / "weights.safetensors")
    for projection in ("w1", "w3", "w2"):
        canonical, scales = weights[projection + "_codes"], weights[projection + "_scales"]
        if canonical.ndim == 3:
            canonical, scales = canonical[0], scales[0]
        codes = pack_w4_nz256(canonical).unsqueeze(0).to(device)
        candidate_codes = pack_int8_layout(codes) if getattr(candidate, "requires_int8_layout", False) else codes
        baseline_codes = pack_int8_layout(codes) if getattr(baseline, "requires_int8_layout", False) else codes
        device_scales = scales.float().unsqueeze(0).to(device)
        reference_weight = unpack_w4(canonical, scales).half().float()
        for rows in (1, 4, 64, 128):
            x_cpu = torch.randn(rows, canonical.shape[1] * 2, generator=torch.Generator().manual_seed(53)).half()
            expected = (x_cpu.float() @ reference_weight.T).half()
            x = x_cpu.to(device)
            ends = torch.tensor([rows], dtype=torch.int64, device=device)
            result = {"projection": projection, "rows": rows, "shape": list(reference_weight.shape), "variants": {}}
            for name, kernel in (("baseline", baseline), ("candidate", candidate)):
                packed = baseline_codes if name == "baseline" else candidate_codes
                actual, timing = timed(partial(kernel, x, packed, device_scales, ends), args.iterations)
                if name == "baseline":
                    baseline_output = actual
                else:
                    result["max_abs_vs_baseline"] = float((actual.float() - baseline_output.float()).abs().max())
                    result["bitwise_equal"] = torch.equal(actual, baseline_output)
                result["variants"][name] = {**timing, "cpu_comparison": metrics(actual, expected)}
            result["speedup"] = (
                result["variants"]["baseline"]["median_seconds"] / result["variants"]["candidate"]["median_seconds"]
            )
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        for kernel in (baseline, candidate):
            if hasattr(kernel, "scales"):
                kernel.scales.clear()
            if hasattr(kernel, "banks"):
                kernel.banks.clear()
        del codes, candidate_codes, baseline_codes, device_scales, x, ends
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
