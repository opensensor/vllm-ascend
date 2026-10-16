# SPDX-License-Identifier: Apache-2.0
"""Paired real-weight grouped MoE gates, excluding shared experts/collectives.

Existing serving workers remain untouched. Timing is discarded if other HTTP
requests overlap. NPU tests require the coherent serving OPP environment.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.qwen4exp.benchmark_int8_projection_probe_310 import server_activity
from tools.qwen4exp.benchmark_prefill_next_310 import load_resources
from tools.qwen4exp.benchmark_w4_prefill_batching_310 import route_geometry
from tools.qwen4exp.native_hybrid_prefill import NativeHybridPrefill
from tools.qwen4exp.native_prefill import NativeGroupedPrefill


def load_grouped_resources(build, device=None, cache_metadata=False, paired_limbs=False):
    resources = load_resources(build)
    manifest = json.loads((build / "provenance.json").read_text())
    entry = manifest["binaries"]["native_grouped_prefill"]
    if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("grouped binary digest mismatch")
    kernel = getattr(torch.classes, manifest["namespace"]).Kernel
    launch = getattr(torch.ops, manifest["namespace"]).launch
    for mode in ("local", "global"):
        resources[mode] = NativeGroupedPrefill(kernel(entry["path"], f"qwen_grouped_{mode}_prefill_v1"), launch)
    if paired_limbs:
        resources["paired"] = NativeGroupedPrefill(kernel(entry["path"], "qwen_grouped_paired_prefill_v1"), launch)
    if cache_metadata:
        resources["cached"] = NativeGroupedPrefill(kernel(entry["path"], "qwen_grouped_cached_prefill_v1"), launch)
    if "native_hybrid_prefill" in manifest["binaries"]:
        entry = manifest["binaries"]["native_hybrid_prefill"]
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("hybrid binary digest mismatch")
        resources["hybrid"] = NativeHybridPrefill(
            kernel(entry["path"], "qwen_hybrid_small_prefill_v1"),
            kernel(entry["path"], "qwen_hybrid_large_prefill_v1"),
            launch,
            device,
        )
        resources["hybrid_a8"] = NativeHybridPrefill(
            kernel(entry["path"], "qwen_hybrid_small_prefill_v1"),
            kernel(entry["path"], "qwen_hybrid_a8_prefill_v1"),
            launch,
            device,
            kernel(entry["path"], "qwen_unpack_a8_prefill_v1"),
        )
    return resources


@torch.inference_mode()
def main():
    """Compare isolated modes on identical routed inputs before timing them.

    Alternate mode order across trials and reject timings overlapped by HTTP
    traffic. Method substitutions affect only this diagnostic's loaded layer;
    the live model, loaded weights, and prefix caches are never mutated here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-limbs", action="store_true", help="requires a binary exporting the paired entry")
    parser.add_argument("--cached-metadata", action="store_true", help="requires a binary exporting the cached entry")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 640, 2560])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists() or min(args.iterations, args.repeats, *args.tokens) <= 0 or max(args.tokens) > 2560:
        parser.error("new output and positive bounded token/timing counts required")
    before = server_activity(args.base_url)
    if before[0]:
        raise RuntimeError("server has user traffic; defer component timing")
    import torch_npu

    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from vllm_ascend.models.qwen4_exp.moe import route_topk
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND
    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(6)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    layer = load_layer(args.model, args.layer, args.rank, 6, NATIVE_INT4_BACKEND)
    resources = load_grouped_resources(args.build, layer.gate.device, args.cached_metadata, args.paired_limbs)
    banks = list(layer.projections.values())
    originals = [bank.native_linear for bank in banks]
    local = SimpleNamespace(gather=resources["route_gather"], swiglu=resources["local_swiglu"])
    cases = []
    generator = torch.Generator().manual_seed(1024)
    for tokens in args.tokens:
        inputs = (torch.randn(tokens, layer.gate.shape[1], generator=generator) * 0.1).half().to(layer.gate.device)
        weights, ids = route_topk(
            torch.nn.functional.linear(inputs, layer.gate),
            layer.top_k,
            renormalize=layer.renormalize,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        cpu_ids = ids.cpu()
        layer.grouped_chunk_tokens = tokens

        def run(mode, inputs=inputs, weights=weights, ids=ids):
            schedule = (
                "paired"
                if mode == "paired_limbs"
                else "cached"
                if mode == "cached_metadata"
                else "hybrid_a8"
                if mode == "hybrid_a8"
                else "hybrid"
                if mode == "hybrid_int8"
                else "local"
                if mode in ("local_schedule", "both")
                else "global"
                if mode == "global_schedule"
                else None
            )
            for bank, original in zip(banks, originals):
                bank.native_linear = (
                    original
                    if schedule is None
                    else (
                        lambda prepared, ends, bank=bank, schedule=schedule: resources[schedule](bank, prepared, ends)
                    )
                )
            if mode in ("local_gather", "both"):
                layer._qwen_local_routes = local
            elif hasattr(layer, "_qwen_local_routes"):
                del layer._qwen_local_routes
            return layer._forward_grouped(inputs, weights, ids)

        reference = run("installed").cpu()
        modes = ("installed", "global_schedule", "local_schedule", "local_gather", "both")
        if "paired" in resources:
            modes += ("paired_limbs",)
        if "cached" in resources:
            modes += ("cached_metadata",)
        if "hybrid" in resources:
            modes += ("hybrid_int8", "hybrid_a8")
        parity = {}
        for mode in modes:
            actual = run(mode).cpu()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            parity[mode] = {"exact_bytes": torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))}
            run(mode)
        torch.npu.synchronize()
        samples = {mode: [] for mode in modes}
        for repeat in range(args.repeats):
            for mode in modes if repeat % 2 == 0 else modes[::-1]:
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    run(mode)
                torch.npu.synchronize()
                samples[mode].append((time.perf_counter() - start) * 1000 / args.iterations)
        cases.append(
            {
                "tokens": tokens,
                **route_geometry(
                    cpu_ids,
                    chunk_tokens=tokens,
                    expert_offset=layer.expert_offset,
                    local_experts=layer.num_local_experts,
                ),
                "modes": {
                    mode: {**parity[mode], "median_ms": statistics.median(samples[mode]), "samples_ms": samples[mode]}
                    for mode in modes
                },
            }
        )
    for bank, original in zip(banks, originals):
        bank.native_linear = original
    after = server_activity(args.base_url)
    if after[0] or after[1] != before[1]:
        raise RuntimeError("user traffic overlapped; discard component timings")
    record = {
        "scope": "one real-weight TP6 partial, fixed router; no shared expert or collective",
        "model": str(args.model),
        "activation_seed": 1024,
        "tp_size": 6,
        "device": args.device,
        "rank": args.rank,
        "layer": args.layer,
        "grouped_activation": layer.grouped_activation,
        "grouped_finalize": layer.grouped_finalize,
        "cases": cases,
        "build": str(args.build),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "server_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
