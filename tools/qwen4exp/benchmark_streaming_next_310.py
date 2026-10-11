# SPDX-License-Identifier: Apache-2.0
"""Standalone gated v2 expert ABBA benchmark. Never contacts the live service."""

import argparse
import json
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

from tools.qwen4exp.build_streaming import verify_bundle
from tools.qwen4exp.thermal_controller import parse_temperatures

STANDALONE_STOP_C = 90
MIN_ABBA_REPEATS = 3
MAX_INPUT_TOKENS = 2560
HIDDEN_SIZE = 2560


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, choices=(4, 6), default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=(128, MAX_INPUT_TOKENS))
    parser.add_argument("--seeds", type=int, nargs="+", default=(1024, 1025))
    parser.add_argument("--repeats", type=int, default=MIN_ABBA_REPEATS)
    parser.add_argument("--expected-devices", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute and an isolated NPU lease are required")
    if (
        args.repeats < MIN_ABBA_REPEATS
        or not 0 <= args.rank < args.tp_size
        or any(not 1 <= n <= MAX_INPUT_TOKENS for n in args.tokens)
    ):
        parser.error("invalid ABBA repeats, rank or token geometry")
    if args.output.exists():
        parser.error("output must be a new append-only evidence path")
    candidate = verify_bundle(args.bundle, require_compiled=True)
    if candidate["configuration"].get("projection_variant") != "m32n160_v2":
        parser.error("a verified v2 projection bundle is required")

    def guard():
        temps = parse_temperatures(
            subprocess.check_output(["npu-smi", "info"], text=True, timeout=3), args.expected_devices
        )
        if max(temps) >= STANDALONE_STOP_C:
            raise RuntimeError(f"standalone temperature stop: {temps}")
        return temps

    guard()
    # Device libraries are loaded only after explicit execution and temperature
    # admission. This is a standalone resource, never a resident server switch.
    import torch
    import torch.nn.functional as functional
    import torch_npu

    from tools.qwen4exp.native_prefill import NativeLocalRouteGather
    from tools.qwen4exp.native_streaming_next import NativeStreamingProjectionNext
    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from tools.qwen4exp.streaming_epilogue import WindowPlan, run_streaming_epilogue
    from tools.qwen4exp.streaming_operands import prepare_grouped_operands
    from vllm_ascend.models.qwen4_exp.moe import route_topk
    from vllm_ascend.models.qwen4_exp.w4_moe import build_grouped_expert_dispatch, finalize_grouped_routes
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND, pack_activation_device
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.ops.load_library(str(args.bundle / candidate["bridges"][0]["path"]))
    kernel = getattr(torch.classes, candidate["namespace"]).Kernel
    launch = getattr(torch.ops, candidate["namespace"]).launch
    binary = args.bundle / "binaries/native_streaming_next.bin"
    projection = NativeStreamingProjectionNext(
        kernel(str(binary), "qwen_streaming_projection_v2"), launch, kernel(str(binary), "qwen_streaming_columns_v2")
    )
    gather = NativeLocalRouteGather(
        kernel(str(args.bundle / "binaries/native_route_gather.bin"), "qwen_local_route_gather_v1"), launch
    )
    layer = load_layer(args.model, 0, args.rank, args.tp_size, NATIVE_INT4_BACKEND)
    layer.grouped_activation, layer.grouped_finalize, layer.grouped_chunk_tokens = (
        "cann_builtin_fp16",
        "cann_v2",
        MAX_INPUT_TOKENS,
    )

    def run_streaming(inputs, weights, ids):
        prepared = prepare_grouped_operands(
            inputs,
            weights,
            ids,
            pack=pack_activation_device,
            dispatch=partial(
                build_grouped_expert_dispatch,
                weight_dtype=layer.compute_dtype,
                count_mode=layer.grouped_route_count_mode,
            ),
            gather=gather,
            num_local_experts=layer.num_local_experts,
            expert_offset=layer.expert_offset,
        )
        projected = projection(layer.projections["gate_up_proj"], prepared.local_operands, prepared.group_ends)
        return run_streaming_epilogue(
            projected,
            prepared.dispatch,
            weights,
            prepared.group_ends,
            layer.projections["down_proj"],
            activation=lambda value: torch_npu.npu_swiglu(value, dim=-1),
            pack=lambda value, _ends: pack_activation_device(value),
            columns=projection.columns,
            finalize=lambda value, dispatch, weights: finalize_grouped_routes(
                value, dispatch, weights, layer.compute_dtype, "cann_v2"
            ),
            complete=lambda *_: None,
            plan=WindowPlan(tile_columns=projection.tile_columns),
        ).output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream, torch.inference_mode():
        for tokens in args.tokens:
            for seed in args.seeds:
                guard()
                inputs = (
                    (torch.randn(tokens, HIDDEN_SIZE, generator=torch.Generator().manual_seed(seed)) * 0.1).half().npu()
                )
                weights, ids = route_topk(
                    functional.linear(inputs, layer.gate),
                    layer.top_k,
                    renormalize=layer.renormalize,
                    routed_scaling_factor=layer.routed_scaling_factor,
                )
                expected, actual = (
                    layer._forward_grouped(inputs, weights, ids).cpu(),
                    run_streaming(inputs, weights, ids).cpu(),
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                trials = {"baseline": [], "streaming": []}
                for _ in range(args.repeats):
                    for name in ("baseline", "streaming", "streaming", "baseline"):
                        guard()
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        result = (
                            layer._forward_grouped(inputs, weights, ids)
                            if name == "baseline"
                            else run_streaming(inputs, weights, ids)
                        )
                        torch.npu.synchronize()
                        trials[name].append((time.perf_counter() - start) * 1000)
                        del result
                record = dict(
                    tokens=tokens,
                    seed=seed,
                    rank=args.rank,
                    tp_size=args.tp_size,
                    exact=True,
                    samples_ms=trials,
                    median_ms={k: statistics.median(v) for k, v in trials.items()},
                    temperatures_c=guard(),
                    candidate_sha256=candidate["candidate_sha256"],
                    gather="native candidate gather; baseline load_layer default gather policy",
                    full_model_validated=False,
                    includes_shared_expert_or_hccl=False,
                )
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
