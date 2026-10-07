# SPDX-License-Identifier: Apache-2.0
"""Measure GLM activation/reduction fusions with alternating paired timings."""

import argparse
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path

from tools.glm_perf.benchmark_route_combine_310 import original_route_combine


def swiglu_reference(gate_up):
    import torch.nn.functional as functional

    gate, up = gate_up.chunk(2, dim=-1)
    return (functional.silu(gate.float()) * up.float()).half()


def measure_pair(torch, original, fused, repeats):
    torch.testing.assert_close(fused().cpu(), original().cpu(), rtol=1e-3, atol=2e-6)
    for _ in range(3):
        original()
        fused()
    torch.npu.synchronize()
    functions = {"original": original, "fused": fused}
    peak, samples = {}, {name: [] for name in functions}
    for name, fn in functions.items():
        torch.npu.reset_peak_memory_stats()
        before = torch.npu.memory_allocated()
        output = fn()
        torch.npu.synchronize()
        peak[name] = torch.npu.max_memory_allocated() - before
        del output
    for repeat in range(repeats):
        names = ("original", "fused") if repeat % 2 == 0 else ("fused", "original")
        for name in names:
            torch.npu.synchronize()
            start = time.perf_counter()
            output = functions[name]()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
            del output
    return {
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        "samples_ms": samples,
        "peak_allocated_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 640, 1280, 2560])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or any(t <= 0 or t > 4096 for t in args.tokens):
        parser.error("output must be new, repeats positive, and tokens in 1..4096")
    import torch
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    if args.library:
        torch.ops.load_library(str(args.library))
    results = []
    for tokens in args.tokens:
        torch.manual_seed(3102026)
        rows = tokens * 8
        gate_up = torch.randn(rows, 4096, dtype=torch.float16, device="npu")
        entry = {"tokens": tokens, "rows": rows, "operation": "swiglu"}
        entry.update(
            measure_pair(
                torch,
                partial(swiglu_reference, gate_up),
                partial(torch.ops._C_ascend.npu_w2_swiglu_310, gate_up),
                args.repeats,
            )
        )
        results.append(entry)
        del gate_up
        routed = torch.randn(rows, 4096, dtype=torch.float16, device="npu")
        routed[rows // 4 :] = 0
        order = torch.randperm(rows).npu()
        inverse = order.argsort()
        weights = torch.rand(tokens, 8, device="npu")
        weights /= weights.sum(1, keepdim=True)
        ends = torch.tensor([rows // 4], dtype=torch.int64, device="npu")
        entry = {"tokens": tokens, "rows": rows, "operation": "route_combine"}
        entry.update(
            measure_pair(
                torch,
                partial(original_route_combine, routed, weights, order, inverse),
                partial(torch.ops._C_ascend.npu_w2_route_combine_310, routed, inverse, weights, ends),
                args.repeats,
            )
        )
        results.append(entry)
        del routed, order, inverse, weights, ends
    document = {
        "cases": results,
        "library": str(args.library),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest() if args.library else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document))


if __name__ == "__main__":
    main()
