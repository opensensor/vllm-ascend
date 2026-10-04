# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen W4 grouped prefill chunk sizes on one real-weight TP partial.

The router is evaluated once. Timings include grouped dispatch, both expert
projections, activation packing, and finalization, but exclude shared experts,
collectives, and the rest of the model. Dry-run only plans route geometry.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

DEFAULT_ROUTE_CAP = 20480
DEFAULT_TOP_K = 10
DEFAULT_CHUNKS = (512, 1024, 2048)


def plan_cases(tokens: int, top_k: int, chunk_sizes: list[int], route_cap: int) -> list[dict]:
    if tokens <= 0 or top_k <= 0 or route_cap <= 0 or not chunk_sizes:
        raise ValueError("tokens, top_k, route_cap, and chunk_sizes must be positive")
    if any(size <= 0 for size in chunk_sizes) or len(set(chunk_sizes)) != len(chunk_sizes):
        raise ValueError("chunk sizes must be positive and distinct")
    return [
        {
            "chunk_tokens": size,
            "projection_calls": (tokens + size - 1) // size,
            "total_projection_calls": 2 * ((tokens + size - 1) // size),
            "largest_call_routes": min(size, tokens) * top_k,
            "within_route_cap": min(size, tokens) * top_k <= route_cap,
        }
        for size in chunk_sizes
    ]


def route_geometry(ids, *, chunk_tokens: int, expert_offset: int, local_experts: int) -> dict:
    """Summarize already-synchronized CPU routes outside the timing window."""
    import torch

    visits = 0
    local_routes = 0
    max_expert_rows = 0
    for start in range(0, ids.shape[0], chunk_tokens):
        local = ids[start : start + chunk_tokens].reshape(-1).to(torch.int64) - expert_offset
        local = local[(local >= 0) & (local < local_experts)]
        counts = torch.bincount(local, minlength=local_experts)
        visits += int((counts > 0).sum())
        local_routes += int(local.numel())
        max_expert_rows = max(max_expert_rows, int(counts.max()))
    return {"local_routes": local_routes, "active_expert_visits": visits, "max_expert_rows": max_expert_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="new directory for separate chunk-size traces")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="dry-run route geometry")
    parser.add_argument("--route-cap", type=int, default=DEFAULT_ROUTE_CAP, help="dry-run route cap")
    parser.add_argument("--chunks", type=int, nargs="+", default=list(DEFAULT_CHUNKS))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.003)
    args = parser.parse_args()
    try:
        cases = plan_cases(args.tokens, args.top_k, args.chunks, args.route_cap)
    except ValueError as exc:
        parser.error(str(exc))
    if args.iterations <= 0 or args.repeats <= 0 or args.atol < 0 or args.rtol < 0:
        parser.error("iterations/repeats must be positive and tolerances nonnegative")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tokens": args.tokens,
                    "top_k": args.top_k,
                    "route_cap": args.route_cap,
                    "cases": cases,
                    "trace_capture": args.trace_dir is not None,
                    "npu_used": False,
                },
                indent=2,
            )
        )
        return
    if (
        args.model is None
        or not args.model.is_dir()
        or args.output is None
        or args.output.exists()
        or (args.trace_dir and args.trace_dir.exists())
    ):
        parser.error("--model must exist and --output/--trace-dir must be new paths")

    import torch
    import torch.nn.functional as F
    import torch_npu

    from tools.qwen4exp.npu_profile import capture_npu_profile
    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from vllm_ascend.models.qwen4_exp.moe import route_topk
    from vllm_ascend.models.qwen4_exp.w4_moe import MAX_GROUPED_NATIVE_ROUTES
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND
    from vllm_ascend.utils import enable_custom_op

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires one Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    with torch.inference_mode():
        layer = load_layer(args.model, args.layer, args.rank, args.tp_size, NATIVE_INT4_BACKEND)
    if args.top_k != layer.top_k or args.route_cap != MAX_GROUPED_NATIVE_ROUTES:
        parser.error("--top-k and --route-cap must match the loaded model and installed Qwen source")
    cases = plan_cases(args.tokens, layer.top_k, args.chunks, MAX_GROUPED_NATIVE_ROUTES)
    if not all(case["within_route_cap"] for case in cases):
        parser.error("chunk exceeds the installed grouped native route cap")
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True)

    generator = torch.Generator().manual_seed(1024)
    with torch.inference_mode():
        inputs = (torch.randn(args.tokens, layer.gate.shape[1], generator=generator) * 0.1).half().npu()
        weights, ids = route_topk(
            F.linear(inputs, layer.gate),
            layer.top_k,
            renormalize=layer.renormalize,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        cpu_ids = ids.cpu()
        layer.grouped_finalize = "torch"

        def run(chunk_tokens: int):
            layer.grouped_chunk_tokens = chunk_tokens
            return layer._forward_grouped(inputs, weights, ids)

        reference = run(max(args.chunks)).cpu().contiguous()
        parity = {}
        for size in args.chunks:
            actual = run(size).cpu().contiguous()
            error = (actual.float() - reference.float()).abs()
            parity[size] = {
                "bitwise_equal": bool(torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))),
                "max_abs": float(error.max()),
                "relative_l2": float(error.norm() / reference.float().norm().clamp_min(1e-12)),
            }
            torch.testing.assert_close(actual, reference, atol=args.atol, rtol=args.rtol)
            for _ in range(2):
                run(size)
        torch.npu.synchronize()
        samples = {size: [] for size in args.chunks}
        for repeat in range(args.repeats):
            order = args.chunks if repeat % 2 == 0 else args.chunks[::-1]
            for size in order:
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    run(size)
                torch.npu.synchronize()
                samples[size].append((time.perf_counter() - start) * 1000 / args.iterations)
        trace_roots = None
        if args.trace_dir:
            trace_roots = {size: str(args.trace_dir / f"chunk-{size}") for size in args.chunks}
            for size in args.chunks:
                capture_npu_profile(lambda size=size: run(size), Path(trace_roots[size]), torch, torch_npu)

    record = {
        "scope": "one real-weight TP partial; precomputed router; grouped dispatch through finalization",
        "model": str(args.model),
        "layer": args.layer,
        "rank": args.rank,
        "tokens": args.tokens,
        "top_k": layer.top_k,
        "route_cap": MAX_GROUPED_NATIVE_ROUTES,
        "route_ids_sha256": hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
        "trace_roots": trace_roots,
        "cases": [
            {
                **case,
                **route_geometry(
                    cpu_ids,
                    chunk_tokens=case["chunk_tokens"],
                    expert_offset=layer.expert_offset,
                    local_experts=layer.num_local_experts,
                ),
                **parity[case["chunk_tokens"]],
                "median_ms": statistics.median(samples[case["chunk_tokens"]]),
                "samples_ms": samples[case["chunk_tokens"]],
            }
            for case in cases
        ],
        "model_ttft_measured": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
