# SPDX-License-Identifier: Apache-2.0
"""Deferred NPU comparison of paired and separate real-weight gate/up launches.

This standalone benchmark opens the selected device. It never calls the serving
API or changes resident weights; run it only when NPU testing is available.
"""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from tools.glm_perf.resident_native import file_digest
from tools.minimax_m3.int8_native import pack_int8_layout
from tools.minimax_m3.int8_native_probe import load_kernel
from tools.minimax_m3.int8_projection_probe import MIN_FREE_BYTES, PROCESS_MEMORY_FRACTION
from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256

WARMUP_CALLS = 3


def timed(function, iterations, *, captured):
    for _ in range(WARMUP_CALLS):
        result = function()
    torch.npu.synchronize()
    graph = None
    if captured:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = function()
        call = graph.replay
    else:
        call = function
    try:
        for _ in range(WARMUP_CALLS):
            call()
        torch.npu.synchronize()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            value = call()
            torch.npu.synchronize()
            samples.append(time.perf_counter() - start)
            if not captured:
                result = value
        return tuple(t.cpu() for t in result), {"median_seconds": statistics.median(samples), "samples": samples}
    finally:
        if graph is not None:
            graph.reset()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    importlib.import_module("torch_npu")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_per_process_memory_fraction(PROCESS_MEMORY_FRACTION, args.device)
    free, _ = torch.npu.mem_get_info()
    if free < MIN_FREE_BYTES:
        raise RuntimeError("isolated paired probe requires at least 1 GiB free")
    candidate, validate = load_kernel(args.candidate)
    baseline, _ = load_kernel(args.baseline)
    if not callable(getattr(candidate, "gate_up", None)) or not baseline.requires_int8_layout:
        raise ValueError("paired candidate and repacked INT8 baseline required")
    validation = validate(candidate)
    if not validation["passed"]:
        raise RuntimeError(f"candidate validation failed: {validation}")
    payload = args.bundle / "weights.safetensors"
    weights = load_file(payload)
    banks, scales = [], []
    for projection in ("w1", "w3"):
        codes, scale = weights[projection + "_codes"], weights[projection + "_scales"]
        if codes.ndim == 3:
            codes, scale = codes[0], scale[0]
        # Two slots with different block scales exercise expert identity; both
        # gate/up matrices come from the real bundle, not random model weights.
        packed = pack_w4_nz256(codes)[None].repeat(2, 1, 1)
        banks.append(pack_int8_layout(packed).npu())
        scales.append(torch.stack((scale, scale * 0.5)).npu())
    reports = []
    for ids in ([0], [1, 0, 1, 0], [2, 1, 0, 2], [2, 2, 2, 2]):
        x = torch.randn(1, banks[0].shape[-1] * 2, generator=torch.Generator().manual_seed(61)).half().npu()
        routes = torch.tensor(ids, dtype=torch.int32, device=x.device)

        def separate(x=x, routes=routes):
            expanded = x.expand(routes.numel(), -1).contiguous()
            return tuple(baseline(expanded, bank, scale, routes) for bank, scale in zip(banks, scales))

        def paired(x=x, routes=routes):
            return candidate.gate_up(x, banks[0], scales[0], banks[1], scales[1], routes)

        for captured in (False, True):
            expected, separate_time = timed(separate, args.iterations, captured=captured)
            actual, paired_time = timed(paired, args.iterations, captured=captured)
            valid = torch.tensor(ids)[:, None] < banks[0].shape[0]
            equal = [torch.equal(torch.where(valid, a, 0), torch.where(valid, b, 0)) for a, b in zip(actual, expected)]
            row = {
                "ids": ids,
                "captured": captured,
                "exact": all(equal),
                "separate": separate_time,
                "paired": paired_time,
                "speedup": separate_time["median_seconds"] / paired_time["median_seconds"],
            }
            reports.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "scope": "Two real expert matrices in two scale-varied slots; isolated projection only, not serving throughput",
        "candidate_sha256": file_digest(args.candidate),
        "baseline_sha256": file_digest(args.baseline),
        "weights_sha256": file_digest(payload),
        "validation": validation,
        "cases": reports,
        "passed": all(row["exact"] for row in reports),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
