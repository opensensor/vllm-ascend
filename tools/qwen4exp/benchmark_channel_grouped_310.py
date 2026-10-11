# SPDX-License-Identifier: Apache-2.0
"""Real routed MoE partial comparison for the lossy whole-K INT8 recipe.

Never modifies serving workers. Includes routing, activation conversion, both
projections, FP16 SwiGLU, and route finalization; excludes shared/collectives.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_int8_projection_probe_310 import server_activity
from tools.qwen4exp.benchmark_int8_requant_projection_310 import requantize_weight
from tools.qwen4exp.native_channel_int8 import (
    NativeChannelActivation,
    NativeChannelInt8,
    NativeDirectChannelActivation,
    NativePackedChannelInt8,
    tile_channel_weights,
)
from tools.qwen4exp.native_hybrid_prefill import signed_nibble_lookup
from tools.qwen4exp.native_prefill import NativeGroupedPrefill

MAX_DIAGNOSTIC_RELATIVE_L2 = 0.1


def channel_bank(bank):
    experts, outputs, half_inputs = bank.weight.shape
    groups = half_inputs // 64
    packed = bank.weight.cpu().reshape(experts, outputs // 16, groups, 2, 16, 32)
    packed = packed.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(experts * outputs, half_inputs)
    fields = []
    for field in (bank.weight_scale, bank.weight_offset):
        decoded = field.cpu().reshape(experts, outputs // 16, groups, 16)
        fields.append(decoded.permute(0, 1, 3, 2).contiguous().reshape(experts * outputs, groups))
    offsets = fields[1].to(torch.int8)
    if not torch.equal(offsets.to(fields[1].dtype), fields[1]):
        raise ValueError("packed channel recipe requires integer offsets in INT8 range")
    dense, quantized, scales = requantize_weight(packed, fields[0], offsets)
    relative_error = float((quantized.float() * scales.unsqueeze(1) - dense).norm() / dense.norm())
    quantized = tile_channel_weights(quantized.reshape(experts, outputs, half_inputs * 2)).to(bank.weight.device)
    scales = scales.reshape(experts, outputs).to(bank.weight.device)
    return quantized, scales, relative_error


@torch.inference_mode()
def main():
    """Gate recipe variants before alternating complete routed-partial timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-activation", action="store_true")
    parser.add_argument("--packed-weight", action="store_true")
    parser.add_argument("--direct-activation", action="store_true")
    parser.add_argument("--column-tile", type=int, choices=(64, 160), default=160)
    parser.add_argument("--row-tile", type=int, choices=(32, 64), default=32)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 640, 2560])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or min(args.iterations, args.repeats, *args.tokens) <= 0 or max(args.tokens) > 2560:
        parser.error("bounded positive timing counts and a new output required")
    if not 0 <= args.rank < 6:
        parser.error("rank must be in [0,5]")
    if args.column_tile == 64 and args.row_tile != 64:
        parser.error("the N64 entry requires M64")
    if args.packed_weight and (not args.direct_activation or args.row_tile != 32 or args.column_tile != 160):
        parser.error("packed expansion requires direct activation and M32/N160")
    if args.local_activation and not args.direct_activation:
        parser.error("local activation requires the direct activation comparison")
    before = server_activity("http://127.0.0.1:8001")
    if before[0]:
        raise RuntimeError("live traffic present")
    import torch_npu

    from tools.qwen4exp.profile_w4_layer_310 import load_layer
    from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
    from vllm_ascend.models.qwen4_exp.moe import route_topk
    from vllm_ascend.models.qwen4_exp.w4_moe import finalize_grouped_routes
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND
    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(6)
    torch.npu.set_device(args.rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    layer = load_layer(args.model, args.layer, args.rank, 6, NATIVE_INT4_BACKEND)
    provenance = json.loads((args.build / "provenance.json").read_text())
    for entry in (provenance["bridge"], *provenance["binaries"].values()):
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("native resource digest mismatch")
    torch.ops.load_library(provenance["bridge"]["path"])
    kernel = getattr(torch.classes, provenance["namespace"]).Kernel
    launch = getattr(torch.ops, provenance["namespace"]).launch
    binary = provenance["binaries"]["native_hybrid_prefill"]["path"]
    projection = NativeChannelInt8(
        kernel(
            binary,
            "qwen_channel_int8_projection_m64_n64_v1"
            if args.column_tile == 64
            else "qwen_channel_int8_projection_m64_v1"
            if args.row_tile == 64
            else "qwen_channel_int8_projection_v1",
        ),
        launch,
        column_tile=args.column_tile,
    )
    activation = NativeChannelActivation(
        kernel(binary, "qwen_channel_activation_v1"), launch, signed_nibble_lookup().to(layer.gate.device)
    )
    compact = NativeGroupedPrefill(
        kernel(provenance["binaries"]["native_grouped_prefill"]["path"], "qwen_grouped_local_prefill_v1"), launch
    )
    direct_activation = (
        NativeDirectChannelActivation(
            kernel(binary, "qwen_direct_channel_activation_v1"),
            launch,
            local_kernel=kernel(binary, "qwen_direct_local_channel_activation_v1") if args.local_activation else None,
        )
        if args.direct_activation
        else None
    )
    packed_projection = (
        NativePackedChannelInt8(
            kernel(binary, "qwen_packed_channel_int8_projection_v1"),
            launch,
            signed_nibble_lookup().to(layer.gate.device),
        )
        if args.packed_weight
        else None
    )
    banks = list(layer.projections.values())
    originals = [bank.native_linear for bank in banks]
    prepared_banks = [channel_bank(bank) for bank in banks]
    generator = torch.Generator().manual_seed(1024)
    records = []
    torch.npu.reset_peak_memory_stats()
    for tokens in args.tokens:
        inputs = (torch.randn(tokens, layer.gate.shape[1], generator=generator) * 0.1).half().to(layer.gate.device)
        weights, ids = route_topk(
            torch.nn.functional.linear(inputs, layer.gate),
            layer.top_k,
            renormalize=layer.renormalize,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        layer.grouped_chunk_tokens = tokens

        def run(mode, inputs=inputs, weights=weights, ids=ids, tokens=tokens):
            if mode in ("direct_int8", "packed_int8", "direct_local_int8"):
                dispatch = build_grouped_expert_dispatch(
                    weights,
                    ids,
                    num_local_experts=layer.num_local_experts,
                    expert_offset=layer.expert_offset,
                    weight_dtype=layer.compute_dtype,
                    count_mode=layer.grouped_route_count_mode,
                )
                order = dispatch.token_indices.index_select(0, dispatch.order)
                ends = dispatch.group_list.contiguous()
                token_codes, token_scales = direct_activation(inputs)

                def project(index, codes, scales):
                    if mode == "packed_int8":
                        return packed_projection(codes, banks[index], prepared_banks[index][1], scales, ends)
                    return projection(codes, prepared_banks[index][0], prepared_banks[index][1], scales, ends)

                projected = project(
                    0,
                    token_codes.index_select(0, order).contiguous(),
                    token_scales.index_select(0, order).contiguous(),
                )
                hidden = torch_npu.npu_swiglu(projected, dim=-1)
                hidden_codes, hidden_scales = direct_activation(hidden, ends if mode == "direct_local_int8" else None)
                output = project(1, hidden_codes, hidden_scales)
                if layer.grouped_finalize == "torch":
                    output = output.to(layer.compute_dtype)
                    output *= dispatch.route_weights.index_select(0, dispatch.order)
                    return output.index_select(0, dispatch.inverse_order).reshape(tokens, layer.top_k, -1).sum(1)
                return finalize_grouped_routes(output, dispatch, weights, layer.compute_dtype, layer.grouped_finalize)
            for bank, original, (channel_weight, channel_scale, _) in zip(banks, originals, prepared_banks):
                if mode == "channel_int8":

                    def channel_linear(prepared, ends, channel_weight=channel_weight, channel_scale=channel_scale):
                        x, sx = activation(prepared)
                        return projection(x, channel_weight, channel_scale, sx, ends)

                    bank.native_linear = channel_linear
                else:
                    bank.native_linear = (
                        original
                        if mode == "installed"
                        else (lambda prepared, ends, bank=bank: compact(bank, prepared, ends))
                    )
            return layer._forward_grouped(inputs, weights, ids)

        reference = run("installed").cpu().float()
        actual = run("channel_int8").cpu().float()
        if not torch.isfinite(actual).all():
            raise RuntimeError("nonfinite grouped whole-K output")
        error = actual - reference
        relative_error = float(error.norm() / reference.norm().clamp_min(1e-12))
        if relative_error > MAX_DIAGNOSTIC_RELATIVE_L2:
            raise RuntimeError(f"grouped numerical gate failed: relative L2={relative_error:.4f}; discard timings")
        compact_result = run("compact").cpu().float()
        torch.testing.assert_close(compact_result, reference, rtol=0, atol=0)
        modes = ("installed", "compact", "channel_int8")
        direct_error = None
        if direct_activation is not None:
            direct = run("direct_int8").cpu().float()
            direct_error = float((direct - reference).norm() / reference.norm().clamp_min(1e-12))
            if not torch.isfinite(direct).all() or direct_error > MAX_DIAGNOSTIC_RELATIVE_L2:
                raise RuntimeError("direct grouped numerical gate failed; discard timings")
            modes += ("direct_int8",)
            if args.local_activation:
                local_result = run("direct_local_int8").cpu().float()
                torch.testing.assert_close(local_result, direct, rtol=0, atol=0)
                modes += ("direct_local_int8",)
            if packed_projection is not None:
                packed_result = run("packed_int8").cpu()
                torch.testing.assert_close(packed_result.float(), direct, rtol=0, atol=0)
                modes += ("packed_int8",)
        for mode in modes:
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
        records.append(
            {
                "tokens": tokens,
                "relative_l2": float(error.norm() / reference.norm().clamp_min(1e-12)),
                "direct_relative_l2": direct_error,
                "max_abs": float(error.abs().max()),
                "exact": torch.equal(actual, reference),
                "timing": {m: {"median_ms": statistics.median(v), "samples_ms": v} for m, v in samples.items()},
            }
        )
    for bank, original in zip(banks, originals):
        bank.native_linear = original
    after = server_activity("http://127.0.0.1:8001")
    if after[0] or after[1] != before[1]:
        raise RuntimeError("traffic overlapped; discard timing")
    record = {
        "row_tile": args.row_tile,
        "column_tile": args.column_tile,
        "rank": args.rank,
        "layer": args.layer,
        "build": str(args.build),
        "server_modified": False,
        "serving_admitted": False,
        "quality_equivalent": False,
        "scope": "real-weight MoE partial with fixed routes; no shared expert or collective",
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        "channel_bank_bytes": sum(b[0].numel() + b[1].numel() * 4 for b in prepared_banks),
        "weight_relative_l2": [b[2] for b in prepared_banks],
        "cases": records,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
