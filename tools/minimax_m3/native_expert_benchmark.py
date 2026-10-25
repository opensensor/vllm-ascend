# SPDX-License-Identifier: Apache-2.0
"""Compare a native candidate with OPP and CPU using full-sized real projections."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from tools.glm_perf.resident_native import NativeSession, file_digest
from tools.minimax_m3.layer_probe import metrics
from tools.minimax_m3.probe_resident_native import operator_exists
from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256, unpack_w4


def main():
    import torch_npu  # noqa: F401 -- CLI-only device registration.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(args.binding))
    session = NativeSession()
    prepared = session.prepare(json.loads(args.manifest.read_text()), operator_exists)
    reports = []
    with torch.inference_mode():
        session.load(prepared["native_digest"], torch.ops.load_library, operator_exists, torch.npu.synchronize)
        native = session.resources[prepared["native_name"]]
        baseline = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
        weights = load_file(args.bundle / "weights.safetensors")
        for projection in ("w1", "w3", "w2"):
            codes, scales = weights[projection + "_codes"], weights[projection + "_scales"]
            reference = unpack_w4(codes, scales).half().float()
            packed = pack_w4_nz256(codes).to(args.device)[None].repeat(64, 1, 1).contiguous()
            device_scales = scales.float().to(args.device)[None].repeat(64, 1, 1).contiguous()
            for rows, boundaries in (
                (1, [1] * 64),
                (2, [1, 2] + [2] * 62),
                (4, [1, 2, 3, 4] + [4] * 60),
                (4, [4] * 64),
                (64, list(range(1, 65))),
                (128, list(range(2, 129, 2))),
                (512, list(range(8, 513, 8))),
                (512, [512] * 64),
                (2048, list(range(32, 2049, 32))),
                (2048, [2048] * 64),
            ):
                x = torch.randn(rows, reference.shape[1], generator=torch.Generator().manual_seed(53)).half()
                expected = (x.float() @ reference.T).half()
                device_x = x.to(args.device)
                ends = torch.tensor(boundaries, dtype=torch.int64, device=args.device)
                variants, original = {}, None
                for name, op in (("opp", baseline), ("native", native)):
                    for _ in range(2):
                        op(device_x, packed, device_scales, ends)
                    torch.npu.synchronize()
                    timings = []
                    for _ in range(args.iterations):
                        start = time.perf_counter()
                        result = op(device_x, packed, device_scales, ends)
                        torch.npu.synchronize()
                        timings.append(time.perf_counter() - start)
                    actual = result.cpu()
                    variants[name] = {
                        "median_seconds": statistics.median(timings),
                        "durations": timings,
                        "cpu_comparison": metrics(actual, expected),
                    }
                    if original is None:
                        original = actual
                    else:
                        variants[name]["opp_comparison"] = metrics(actual, original)
                active_experts = sum(end > start for start, end in zip([0, *boundaries[:-1]], boundaries))
                record = {
                    "projection": projection,
                    "rows": rows,
                    "active_experts": active_experts,
                    "variants": variants,
                }
                reports.append(record)
                print(json.dumps(record), flush=True)
            del packed, device_scales
    report = {
        "scope": "One real expert repeated in 64 slots; projection only, not serving throughput",
        "manifest": str(args.manifest),
        "manifest_sha256": file_digest(args.manifest),
        "weights_sha256": file_digest(args.bundle / "weights.safetensors"),
        "binding_sha256": file_digest(args.binding),
        "cases": reports,
        "passed": all(
            v["cpu_comparison"]["finite"] and v["cpu_comparison"]["nrmse"] < 0.001
            for r in reports
            for v in r["variants"].values()
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
