# SPDX-License-Identifier: Apache-2.0
"""Gate fused SwiGLU packing in one real-weight Qwen W4 MoE TP partial.

Router inputs stay fixed. Timings include grouped dispatch, both native W4
projections, activation preparation, and route finalization. They exclude
attention, shared experts, collectives, and service scheduling.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

DEFAULT_TOKENS = 1536


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.tokens <= 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("tokens, iterations, and repeats must be positive")
    if args.dry_run:
        print(json.dumps({"tokens": args.tokens, "arms": ["torch", "cann_swiglu_pack"], "npu_used": False}))
        return
    if args.model is None or not args.model.is_dir() or args.output is None or args.output.exists():
        parser.error("--model must exist and --output must be a new path")

    import torch
    import torch.nn.functional as F
    import torch_npu

    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from vllm_ascend.models.qwen4_exp.moe import route_topk
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND
    from vllm_ascend.utils import enable_custom_op

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires one Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    with torch.inference_mode():
        layer = load_layer(args.model, args.layer, args.rank, args.tp_size, NATIVE_INT4_BACKEND)
        if args.tokens > layer.grouped_chunk_tokens:
            parser.error("tokens exceed the model's selected grouped chunk")
        generator = torch.Generator().manual_seed(1024)
        inputs = (torch.randn(args.tokens, layer.gate.shape[1], generator=generator) * 0.1).half().npu()
        weights, ids = route_topk(
            F.linear(inputs, layer.gate),
            layer.top_k,
            renormalize=layer.renormalize,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

        def run(method: str):
            layer.grouped_activation = method
            return layer._forward_grouped(inputs, weights, ids)

        expected = run("torch").cpu().contiguous()
        actual = run("cann_swiglu_pack").cpu().contiguous()
        difference = (actual.float() - expected.float()).abs()
        bitwise_equal = bool(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        samples = {"torch": [], "cann_swiglu_pack": []}
        if bitwise_equal:
            for method in samples:
                for _ in range(2):
                    run(method)
            torch.npu.synchronize()
            for repeat in range(args.repeats):
                order = tuple(samples) if repeat % 2 == 0 else tuple(reversed(samples))
                for method in order:
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        run(method)
                    torch.npu.synchronize()
                    samples[method].append((time.perf_counter() - start) * 1000 / args.iterations)

    record = {
        "scope": "one real-weight TP partial; precomputed router; grouped dispatch through finalization",
        "model": str(args.model),
        "layer": args.layer,
        "rank": args.rank,
        "tokens": args.tokens,
        "top_k": layer.top_k,
        "route_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        "reference_output_sha256": hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
        "candidate_output_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
        "bitwise_equal": bitwise_equal,
        "max_abs": float(difference.max()),
        "timings_ms": {
            method: {"median_ms": statistics.median(values), "samples_ms": values} if values else None
            for method, values in samples.items()
        },
        "model_ttft_measured": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if not bitwise_equal:
        raise AssertionError("fused prefill SwiGLU pack changed real-weight layer output")


if __name__ == "__main__":
    main()
