# SPDX-License-Identifier: Apache-2.0
"""Check prepared INT8 projections against real W4 weights and ordered G128 math.

This isolated probe is not a serving backend. Run hardware validation only in
a controlled idle window. Prepared operand timing excludes quantization,
routing, and weight expansion, and cannot establish whole-model throughput.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch
from safetensors import safe_open

GROUP_SIZE = 128
OUTPUT_BLOCK = 16
CUBE_K_BLOCK = 32
SCALE_LANES = 8
MAX_INPUTS = 2560
OUTPUT_TILE = 160


def server_activity(base_url):
    """Read gauges/counter only; never pause or mutate the live server."""
    with urlopen(base_url.rstrip("/") + "/metrics", timeout=5) as response:
        metrics = response.read().decode()
    active = 0.0
    completed = 0.0
    seen = set()
    for line in metrics.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            active += float(line.rsplit(" ", 1)[-1])
            seen.add(line.split("{", 1)[0])
        elif line.startswith("vllm:request_success_total{"):
            completed += float(line.rsplit(" ", 1)[-1])
            seen.add("vllm:request_success_total")
    if seen != {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_success_total"}:
        raise RuntimeError("server metrics do not expose the required timing guard gauges and counter")
    return active, completed


def prepare_weights(packed, scale, offset):
    """Expand q-offset losslessly, retaining each checkpoint G128 FP16 scale."""
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError("expected a packed INT8 matrix")
    outputs, half_inputs = packed.shape
    inputs = half_inputs * 2
    groups = inputs // GROUP_SIZE
    if not 0 < inputs <= MAX_INPUTS or inputs % GROUP_SIZE or outputs % OUTPUT_TILE:
        raise ValueError("unsupported projection geometry")
    if (
        scale.shape != (outputs, groups)
        or offset.shape != scale.shape
        or scale.dtype != torch.float16
        or offset.dtype != torch.int8
    ):
        raise ValueError("invalid G128 checkpoint metadata")
    nibbles = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).flatten(-2)
    codes = torch.where(nibbles >= 8, nibbles - 16, nibbles).int().reshape(outputs, groups, GROUP_SIZE)
    centered = codes - offset.int().unsqueeze(-1)
    if centered.min() < -128 or centered.max() > 127:
        raise ValueError("q-offset does not fit lossless INT8")
    centered = centered.to(torch.int8).reshape(outputs, inputs)
    tiled = (
        centered.reshape(outputs // OUTPUT_BLOCK, OUTPUT_BLOCK, groups, GROUP_SIZE // CUBE_K_BLOCK, CUBE_K_BLOCK)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    scales = scale.reshape(outputs // OUTPUT_BLOCK, OUTPUT_BLOCK, groups).transpose(1, 2).contiguous()
    return centered, tiled, scales


def prepare_activation(activation):
    if activation.dtype != torch.float16 or activation.ndim != 2 or activation.shape[1] % GROUP_SIZE:
        raise ValueError("expected FP16 rows with complete G128 groups")
    grouped = activation.float().reshape(activation.shape[0], -1, GROUP_SIZE)
    maximum = grouped.abs().amax(-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    quantized = (grouped / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape_as(activation), scale.unsqueeze(-1).expand(-1, -1, SCALE_LANES).contiguous()


def ordered_reference(activation, centered, weight_scale, activation_scale):
    result = torch.zeros((activation.shape[0], centered.shape[0]), dtype=torch.float32)
    for group in range(activation.shape[1] // GROUP_SIZE):
        first = group * GROUP_SIZE
        dot = activation[:, first : first + GROUP_SIZE].int() @ centered[:, first : first + GROUP_SIZE].int().t()
        product = dot.float() * weight_scale[:, group].float().unsqueeze(0)
        product *= activation_scale[:, group, 0].unsqueeze(1)
        result += product
    return result.half()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 3, 16, 17])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="read-only traffic guard for timings")
    parser.add_argument("--fused-quant", action="store_true", help="run direct A8 quantization before projection")
    parser.add_argument(
        "--compare-w4", action="store_true", help="also check/time the prepared native two-limb operator"
    )
    parser.add_argument("--cpu-only", action="store_true", help="inspect real weights without opening any NPU")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier receipts")
    manifest = json.loads((args.build_dir / "provenance.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((args.build_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"probe digest mismatch: {name}")
    if not args.cpu_only:
        # Lazy hardware imports keep the offline path independent of torch-npu.
        import torch_npu

        from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms

        torch.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.ops.load_library(manifest["bridge"])
        kernel = torch.classes.qwen_int8_group_probe_v1.Kernel(
            manifest["projection_binary"], "qwen_int8_projection_probe_v1"
        )
        quantizer = None
        if args.fused_quant:
            quantizer = torch.classes.qwen_int8_group_probe_v1.Kernel(
                manifest["activation_binary"], "qwen_int8_activation_probe_v1"
            )
        if args.compare_w4:
            from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
                pack_activation_device,
                pack_native_metadata,
                pack_native_weight,
                quantize_activation_limbs,
            )
            from vllm_ascend.utils import enable_custom_op

            enable_custom_op()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    generator = torch.Generator().manual_seed(314)
    records = []
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        prefix = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}.{projection}"
        fields = []
        for kind in ["weight", "weight_scale", "weight_offset"]:
            name = f"{prefix}.{kind}"
            with safe_open(args.model / index[name], framework="pt", device="cpu") as shard:
                fields.append(shard.get_tensor(name))
        centered, tiled, scales = prepare_weights(*fields)
        if not args.cpu_only:
            weights, device_scales = tiled.npu(), scales.npu()
            if args.compare_w4:
                native_weight, sums = pack_native_weight(fields[0])
                native = tuple(
                    value.unsqueeze(0).npu()
                    for value in (
                        native_weight,
                        pack_native_metadata(fields[1]),
                        pack_native_metadata(fields[2]).half(),
                        sums,
                    )
                )
        for rows in args.tokens:
            activation = (torch.randn(rows, centered.shape[1], generator=generator) * 0.1).half()
            quantized, activation_scales = prepare_activation(activation)
            reference = ordered_reference(quantized, centered, fields[1], activation_scales)
            if args.cpu_only:
                assert torch.isfinite(reference).all()
                records.append(
                    {
                        "projection": projection,
                        "rows": rows,
                        "shape": list(centered.shape),
                        "weight_layout_roundtrip": bool(
                            torch.equal(tiled.permute(0, 3, 1, 2, 4).reshape_as(centered), centered)
                        ),
                        "scale_layout_roundtrip": bool(
                            torch.equal(scales.transpose(1, 2).reshape_as(fields[1]), fields[1])
                        ),
                        "maximum_centered_code": int(centered.abs().max()),
                        "weight_bytes": tiled.numel(),
                        "packed_w4_bytes": fields[0].numel(),
                        "hardware_validated": False,
                    }
                )
                continue
            x, sx = quantized.npu(), activation_scales.npu()
            output = torch.empty((rows, centered.shape[0]), dtype=torch.float16, device="npu")
            descriptor = torch.tensor([rows, centered.shape[1], centered.shape[0]], dtype=torch.int64, device="npu")
            quant_operands = ()
            quant_blocks = 1
            if args.fused_quant:
                source = activation.npu()
                quant_descriptor = torch.tensor(
                    [rows * centered.shape[1] // GROUP_SIZE], dtype=torch.int64, device="npu"
                )
                quant_operands = (source, x, sx, quant_descriptor)
                quant_blocks = min(8, (rows * centered.shape[1] // GROUP_SIZE + 7) // 8)

            blocks = min(8, centered.shape[0] // OUTPUT_TILE)

            def run(
                kernel=kernel,
                operands=(x, weights, device_scales, sx, output, descriptor),
                blocks=blocks,
                quantizer=quantizer,
                quant_operands=quant_operands,
                quant_blocks=quant_blocks,
            ):
                if quantizer is not None:
                    torch.ops.qwen_int8_group_probe_v1.launch(quantizer, list(quant_operands), quant_blocks)
                torch.ops.qwen_int8_group_probe_v1.launch(kernel, list(operands), blocks)

            run()
            torch.npu.synchronize()
            actual = output.cpu()
            if args.fused_quant:
                assert torch.equal(x.cpu(), quantized)
                torch.testing.assert_close(sx.cpu(), activation_scales, rtol=0, atol=0)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                run()
            torch.npu.current_stream().wait_stream(stream)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                run()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
            record = {
                "projection": projection,
                "rows": rows,
                "shape": list(centered.shape),
                "exact_ordered_fp16": True,
                "graph_replay": True,
                "weight_bytes": weights.numel(),
                "packed_w4_bytes": fields[0].numel(),
                "includes_direct_a8_quantization": args.fused_quant,
            }
            activity = server_activity(args.base_url) if args.time else None
            measure = args.time and activity[0] == 0
            if args.time and not measure:
                record["timings_deferred"] = "live_requests_active"
            if measure:
                prefix = "quant_and_projection" if args.fused_quant else "prepared"
                record[f"{prefix}_eager"] = timed_ms(run, 30, 3)
                record[f"{prefix}_graph"] = timed_ms(graph.replay, 30, 3)
            if args.compare_w4:
                limbs = tuple(value.npu() for value in quantize_activation_limbs(activation))
                ends = torch.tensor([rows], dtype=torch.int64, device="npu")

                def baseline(
                    limbs=limbs, native=native, ends=ends, source=quant_operands[0] if quant_operands else None
                ):
                    prepared = pack_activation_device(source) if source is not None else limbs
                    return torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(*prepared, *native, ends)

                torch.testing.assert_close(baseline().cpu(), reference, rtol=0, atol=0)
                record["native_w4_exact"] = True
                if measure:
                    prefix = "native_w4_quant_and_projection" if args.fused_quant else "native_w4_prepared"
                    record[f"{prefix}_eager"] = timed_ms(baseline, 30, 3)
                    baseline_graph = torch.npu.NPUGraph()
                    with torch.npu.graph(baseline_graph, stream=stream):
                        baseline_output = baseline()
                    baseline_graph.replay()
                    torch.testing.assert_close(baseline_output.cpu(), reference, rtol=0, atol=0)
                    record[f"{prefix}_graph"] = timed_ms(baseline_graph.replay, 30, 3)
                    del baseline_graph, baseline_output
            if measure:
                after = server_activity(args.base_url)
                if after[0] != 0 or after[1] != activity[1]:
                    for key in list(record):
                        if key.endswith(("_eager", "_graph")):
                            del record[key]
                    record["timings_discarded"] = "live_requests_overlapped"
                else:
                    record["traffic_guard"] = "idle_before_and_after_no_completed_request"
            records.append(record)
            print(json.dumps(record), flush=True)
            del graph
    args.output.write_text(
        json.dumps(
            {
                "passed": True,
                "hardware_validated": not args.cpu_only,
                "serving_speedup_proven": False,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
