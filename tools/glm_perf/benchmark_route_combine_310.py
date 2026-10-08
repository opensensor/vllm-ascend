# SPDX-License-Identifier: Apache-2.0
"""Compare weighted inverse-sort/reduction with its FP32 fused 310P operator."""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path


def original_route_combine(routed, weights, order, inverse):
    expanded = routed.float()
    expanded *= weights.reshape(-1, 1).index_select(0, order)
    return expanded.index_select(0, inverse).reshape(*weights.shape, routed.shape[1]).sum(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 640, 1280])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(tokens <= 0 or tokens * 8 > 32768 for tokens in args.tokens) or args.repeats < 1:
        parser.error("tokens must be in 1..4096 and repeats must be positive")
    if args.output.exists():
        parser.error("output already exists")

    # This is an explicit hardware entry point; --help never initializes NPU.
    import torch
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    combine = torch.ops._C_ascend.npu_w2_route_combine_310
    results = []
    for tokens in args.tokens:
        torch.manual_seed(3102026)
        rows, hidden, top_k = tokens * 8, 4096, 8
        local_rows = rows // 4
        routed = torch.randn(rows, hidden, dtype=torch.float16)
        routed[local_rows:] = 0
        order = torch.randperm(rows)
        inverse = torch.argsort(order).npu()
        weights = torch.rand(tokens, top_k)
        weights /= weights.sum(1, keepdim=True)
        routed, weights, order = routed.npu(), weights.npu(), order.npu()
        ends = torch.tensor([local_rows], dtype=torch.int64, device="npu")

        original = partial(original_route_combine, routed, weights, order, inverse)
        fused = partial(combine, routed, inverse, weights, ends)

        torch.testing.assert_close(fused(), original(), rtol=2e-6, atol=2e-6)
        for _ in range(3):
            original()
            fused()
        torch.npu.synchronize()
        samples = {"original": [], "fused": []}
        peak_bytes = {}
        for name, fn in (("original", original), ("fused", fused)):
            torch.npu.reset_peak_memory_stats()
            before = torch.npu.memory_allocated()
            output = fn()
            torch.npu.synchronize()
            peak_bytes[name] = torch.npu.max_memory_allocated() - before
            del output
        for repeat in range(args.repeats):
            order_names = ("original", "fused") if repeat % 2 == 0 else ("fused", "original")
            for name in order_names:
                torch.npu.synchronize()
                start = time.perf_counter()
                output = original() if name == "original" else fused()
                torch.npu.synchronize()
                samples[name].append(1000 * (time.perf_counter() - start))
                del output
        results.append(
            {
                "tokens": tokens,
                "local_rows": local_rows,
                "samples_ms": samples,
                "median_ms": {key: statistics.median(value) for key, value in samples.items()},
                "peak_allocated_bytes": peak_bytes,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": results}, indent=2) + "\n")
    print(json.dumps({"cases": results}))


if __name__ == "__main__":
    main()
