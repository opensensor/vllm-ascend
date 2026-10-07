# SPDX-License-Identifier: Apache-2.0
"""Isolate the long-prefill selection boundary without loading GLM.

Run one shape per process on released hardware. Start with --stage topk,
then --stage score-select. --trace synchronizes each ATen/custom operation;
also reproduce without it, because synchronization can hide lifetime bugs.
"""

import argparse
import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from tools.glm_perf.trace_device_ops import DeviceOpTrace
from vllm_ascend.models.glm5next import kpool_ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=640)
    parser.add_argument("--pools", type=int, default=4320)
    parser.add_argument("--stage", choices=("topk", "score-select"), default="topk")
    parser.add_argument("--rows-per-call", type=int)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.rows <= 0 or args.pools < 512 or args.repeats <= 0:
        parser.error("rows/repeats must be positive and pools must be >= 512")
    if args.rows_per_call is not None and args.rows_per_call <= 0:
        parser.error("rows-per-call must be positive")
    if args.stage == "score-select" and args.pools * 4 - args.rows + 1 < 2048:
        parser.error("score-select requires at least 512 completed pools for every query")
    # Lazy import: CPU tests and --help must not load or initialize the NPU.
    import torch_npu

    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310)
    if args.stage == "topk":
        # Unique FP32 scores make CPU index parity unambiguous.
        scores_cpu = torch.stack([torch.randperm(args.pools, generator=generator) for _ in range(args.rows)]).float()
        scores = scores_cpu.npu()
        expected = scores_cpu.topk(512, dim=1).indices.int()

        def operation():
            return kpool_ops.topk_pool_indices(scores, 512, rows_per_call=args.rows_per_call)

    else:
        q, weights, keys = [
            value.npu()
            for value in (
                torch.randn(args.rows, 32, 128, generator=generator).half(),
                torch.randn(args.rows, 32, generator=generator),
                torch.randn(args.pools, 128, generator=generator).bfloat16(),
            )
        ]
        positions = torch.arange(args.pools * 4 - args.rows, args.pools * 4, dtype=torch.int32).npu()
        expected = None
        if args.rows_per_call is not None:
            kpool_ops.topk_pool_indices = partial(kpool_ops.topk_pool_indices, rows_per_call=args.rows_per_call)

        def operation():
            return kpool_ops.score_and_select_kpool_tokens(q, weights, keys, positions, 2048, 4)

    torch.npu.synchronize()
    with args.trace.open("w") if args.trace else nullcontext(None) as output:
        for repeat in range(args.repeats):
            print(
                json.dumps(
                    {
                        "event": "begin",
                        "repeat": repeat,
                        "stage": args.stage,
                        "rows": args.rows,
                        "pools": args.pools,
                        "rows_per_call": args.rows_per_call,
                    }
                ),
                flush=True,
            )
            started = perf_counter()
            with DeviceOpTrace(output, torch.npu.synchronize) if output else nullcontext():
                actual = operation()
                torch.npu.synchronize()
            elapsed = perf_counter() - started
            actual = actual.cpu()
            if expected is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                # Selection must contain 512 distinct causal pools plus tail.
                groups = actual[:, :2048:4] // 4
                completed = (positions.cpu() + 1) // 4
                assert ((groups >= 0) & (groups < completed[:, None])).all()
                assert (groups.sort(dim=1).values.diff(dim=1) > 0).all()
            print(
                json.dumps(
                    {
                        "event": "end",
                        "repeat": repeat,
                        "seconds": elapsed,
                        "check": "CPU exact indices" if expected is not None else "causal distinct pools",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
