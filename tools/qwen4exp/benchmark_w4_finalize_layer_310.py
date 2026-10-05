# SPDX-License-Identifier: Apache-2.0
"""Compare grouped MoE finalizers on one real-weight Qwen W4 TP partial.

Inputs are synthetic activations. This checks one layer, not model quality or
end-to-end cold-prefill latency.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

TOKENS = (512, 2048)
METHODS = ("torch", "cann_v2")


def main():
    """Load one real-weight layer and compare opt-in finalizers on a 310P."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.003)
    parser.add_argument("--trace-dir", type=Path, help="new directory for traces of the largest token shape")
    args = parser.parse_args()
    if args.iterations <= 0 or args.trials <= 0 or any(tokens <= 0 for tokens in args.tokens):
        parser.error("iterations, trials, and token counts must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be nonnegative")
    if args.trace_dir and len(set(args.tokens)) != len(args.tokens):
        parser.error("trace capture requires unique token counts")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tokens": args.tokens,
                    "methods": METHODS,
                    "trace_capture": args.trace_dir is not None,
                    "npu_used": False,
                }
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
        parser.error("--model must exist and --output/--trace-dir must name new paths")

    import torch
    import torch.nn.functional as F
    import torch_npu

    from tools.qwen4exp.benchmark_w4_finalize_routing_310 import capture_npu_profile
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
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/models/qwen4_exp/w4_moe.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    generator = torch.Generator().manual_seed(1024)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True)

    with torch.inference_mode(), args.output.open("x") as output:
        for tokens in args.tokens:
            inputs = (torch.randn(tokens, layer.gate.shape[1], generator=generator) * 0.1).half().npu()
            weights, ids = route_topk(
                F.linear(inputs, layer.gate),
                layer.top_k,
                renormalize=layer.renormalize,
                routed_scaling_factor=layer.routed_scaling_factor,
            )

            def forward(method, inputs=inputs, weights=weights, ids=ids):
                layer.grouped_finalize = method
                return layer._forward_grouped(inputs, weights, ids)

            reference = forward("torch").cpu().float()
            candidate = forward("cann_v2").cpu().float()
            difference = (candidate - reference).abs()
            max_abs = float(difference.max())
            relative_l2 = float(difference.norm() / reference.norm().clamp_min(1e-12))
            torch.testing.assert_close(candidate, reference, atol=args.atol, rtol=args.rtol)

            for method in METHODS:
                for _ in range(2):
                    forward(method)
            torch.npu.synchronize()
            samples = {method: [] for method in METHODS}
            for trial in range(args.trials):
                order = METHODS if trial % 2 == 0 else METHODS[::-1]
                for method in order:
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        forward(method)
                    torch.npu.synchronize()
                    samples[method].append((time.perf_counter() - start) * 1000 / args.iterations)
            trace_roots = None
            if args.trace_dir and tokens == max(args.tokens):
                trace_roots = {method: str(args.trace_dir / method) for method in METHODS}
                for method in METHODS:
                    capture_npu_profile(
                        lambda method=method: forward(method), Path(trace_roots[method]), torch, torch_npu
                    )
            record = {
                "tokens": tokens,
                "layer": args.layer,
                "rank": args.rank,
                "tp_size": args.tp_size,
                "weights": "real_checkpoint",
                "activations": "synthetic_seed_1024_std_0.1",
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "timings_ms": {
                    method: {"median_ms": statistics.median(values), "trials_ms": values}
                    for method, values in samples.items()
                },
                "source_sha256": source_hash,
                "trace_roots": trace_roots,
                "model_ttft_measured": False,
            }
            line = json.dumps(record)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
