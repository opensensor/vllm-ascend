# SPDX-License-Identifier: Apache-2.0
"""Measure repeated expert projection across token chunks on identical routes.

Planning is CPU-testable. Device timing includes only grouped projections;
routing, input gathers, and final output reordering are prepared outside it.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RouteChunk:
    begin: int
    end: int
    order: torch.Tensor
    group_ends: torch.Tensor


def plan_chunks(ids: torch.Tensor, experts: int, chunk_tokens: int) -> list[RouteChunk]:
    """Group each token chunk; -1 represents peer-owned routes."""
    if ids.device.type != "cpu" or ids.ndim != 2 or ids.dtype != torch.int64:
        raise ValueError("ids must be a CPU int64 [tokens, top_k] tensor")
    if experts <= 0 or chunk_tokens <= 0 or ids.shape[1] == 0:
        raise ValueError("experts, chunk_tokens, and top_k must be positive")
    if ((ids < -1) | (ids >= experts)).any():
        raise ValueError("expert ids must be -1 or a local expert index")
    chunks = []
    top_k = ids.shape[1]
    for first in range(0, ids.shape[0], chunk_tokens):
        begin, end = first * top_k, min(first + chunk_tokens, ids.shape[0]) * top_k
        local = ids.reshape(-1)[begin:end]
        order = torch.argsort(torch.where(local < 0, experts, local), stable=True)
        ends = torch.bincount(local[local >= 0], minlength=experts).cumsum(0)
        chunks.append(RouteChunk(begin, end, order, ends))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=640)
    parser.add_argument("--max-routes", type=int, default=5120, help="explicit installed OPP route cap")
    parser.add_argument("--chunk-tokens", type=int, nargs="+", default=[64, 128, 256, 640])
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--projection", choices=["gate_up", "down"], default="gate_up")
    parser.add_argument("--routing", choices=["uniform", "hot"], default="uniform")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.tokens <= args.max_routes // 8 or min(args.chunk_tokens) <= 0 or args.repeats <= 0:
        parser.error("tokens must fit --max-routes / 8; chunk sizes and repeats must be positive")

    # Import the device stack only for an explicitly invoked hardware run.
    import torch_npu

    from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    experts, top_k, n = 72, 8, 4096
    k = 4096 if args.projection == "gate_up" else 2048
    gen = torch.Generator().manual_seed(310640)
    # Distinct top-8 selections from 288 experts. One rank owns experts 0..71.
    scores = torch.rand(args.tokens, experts * 4, generator=gen)
    if args.routing == "hot":
        scores[:, :8] += 2
    ids = scores.topk(top_k, dim=-1).indices
    ids = torch.where(ids < experts, ids, -1)
    codes = torch.randint(0, 256, (experts, n, k * args.bits // 8), generator=gen, dtype=torch.uint8)
    pack = _pack_codes_nz_w3 if args.bits == 3 else _pack_codes_nz
    packed = torch.stack([pack(codes[e], k) for e in range(experts)]).view(torch.int8).npu()
    scales = (torch.rand(experts, n // 32, k // 32, generator=gen) * 0.02 + 0.005).npu()
    inputs = torch.randn(args.tokens, k, generator=gen).half().repeat_interleave(top_k, dim=0)
    sizes = sorted(set([*args.chunk_tokens, args.tokens]))
    prepared = {}
    for size in sizes:
        prepared[size] = [
            (chunk, inputs[chunk.begin : chunk.end][chunk.order].npu(), chunk.group_ends.npu())
            for chunk in plan_chunks(ids, experts, size)
        ]

    def run(size):
        return [op(x, packed, scales, ends) for _, x, ends in prepared[size]]

    def restore(size, outputs):
        restored = torch.empty(args.tokens * top_k, n, dtype=torch.float16)
        for (chunk, _, _), output in zip(prepared[size], outputs):
            restored[chunk.begin + chunk.order] = output.cpu()
        return restored

    reference = restore(args.tokens, run(args.tokens))
    for size in sizes:
        actual = restore(size, run(size))
        if not torch.isfinite(actual).all() or not torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)):
            raise AssertionError(f"{size}-token chunks differ from the single grouped call")
        run(size)
    torch.npu.synchronize()
    samples = {size: [] for size in sizes}
    for repeat in range(args.repeats):
        for size in sizes if repeat % 2 == 0 else sizes[::-1]:
            torch.npu.synchronize()
            start = time.perf_counter()
            outputs = run(size)
            torch.npu.synchronize()
            samples[size].append((time.perf_counter() - start) * 1000)
            del outputs
    result = {
        "scope": "projection calls only; prepared routing and gathers; same weights and routes",
        "bits": args.bits,
        "projection": args.projection,
        "routing": args.routing,
        "tokens": args.tokens,
        "local_routes": int((ids >= 0).sum()),
        "cases": [
            {
                "chunk_tokens": size,
                "calls": len(prepared[size]),
                "bitwise_equal": True,
                "median_ms": statistics.median(samples[size]),
                "samples_ms": samples[size],
            }
            for size in sizes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
