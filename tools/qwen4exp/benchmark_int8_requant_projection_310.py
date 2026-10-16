# SPDX-License-Identifier: Apache-2.0
"""Measure whole-K INT8 speed and error using real G128 checkpoint weights.

This changes both activation and weight quantization. It is an isolated math
experiment, never a serving backend or evidence of language-quality parity.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.qwen4exp.benchmark_int8_projection_probe_310 import prepare_weights, server_activity
from tools.qwen4exp.profile_projection_dtypes_310 import timed

GROUP_SIZE = 128
NZ_FORMAT = 29


def requantize_weight(packed, scale, offset):
    """Fold qualified per-group scales into a new per-output INT8 weight."""
    from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import quantize_linear_weight

    centered, _, _ = prepare_weights(packed, scale, offset)
    dense = centered.float() * scale.float().repeat_interleave(GROUP_SIZE, dim=1)
    quantized, channel_scale = quantize_linear_weight(dense)
    return dense, quantized, channel_scale


@torch.inference_mode()
def main():
    """Compare the installed G128 projection with explicitly lossy full-K math."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, help="optional native whole-K build")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[3, 128, 512])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists() or min(args.iterations, args.repeats, *args.tokens) <= 0 or max(args.tokens) > 2560:
        parser.error("positive bounded timing counts and new output required")
    before = server_activity(args.base_url)
    if before[0]:
        raise RuntimeError("server traffic present; defer hardware math experiment")
    import torch_npu

    from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import dynamic_w8a8_linear
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device, pack_native_metadata, pack_native_weight
    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(6)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    resource = None
    if args.build:
        from tools.qwen4exp.native_channel_int8 import NativeChannelInt8, tile_channel_weights

        manifest = json.loads((args.build / "provenance.json").read_text())
        for entry in (manifest["bridge"], manifest["binaries"]["native_hybrid_prefill"]):
            if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("whole-K build digest mismatch")
        torch.ops.load_library(manifest["bridge"]["path"])
        kernel = getattr(torch.classes, manifest["namespace"]).Kernel
        resource = NativeChannelInt8(
            kernel(manifest["binaries"]["native_hybrid_prefill"]["path"], "qwen_channel_int8_projection_v1"),
            getattr(torch.ops, manifest["namespace"]).launch,
        )
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    generator = torch.Generator().manual_seed(314)
    records = []
    for projection, parts in (("gate_up", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
        fields = []
        for kind in ("weight", "weight_scale", "weight_offset"):
            tensors = []
            for part in parts:
                name = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}.{part}.{kind}"
                with safe_open(args.model / index[name], framework="pt", device="cpu") as shard:
                    tensors.append(shard.get_tensor(name))
            fields.append(torch.cat(tensors))
        dense, quantized, channel_scale = requantize_weight(*fields)
        requantized = quantized.float() * channel_scale.unsqueeze(1)
        weight_error = float((requantized - dense).norm() / dense.norm())
        runtime_weight = torch_npu.npu_format_cast(quantized.npu(), NZ_FORMAT).t()
        runtime_scale = channel_scale.npu()
        if resource:
            channel_bank = tile_channel_weights(quantized.unsqueeze(0)).npu()
            channel_scales = channel_scale.unsqueeze(0).npu()
        packed, total = pack_native_weight(fields[0])
        native = tuple(
            t.unsqueeze(0).npu()
            for t in (packed, pack_native_metadata(fields[1]), pack_native_metadata(fields[2]).half(), total)
        )
        for rows in args.tokens:
            x = (torch.randn(rows, dense.shape[1], generator=generator) * 0.1).half().npu()
            ends = torch.tensor([rows], dtype=torch.int64, device=x.device)

            def original(x=x, native=native, ends=ends):
                return torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(*pack_activation_device(x), *native, ends)

            def candidate(x=x, runtime_weight=runtime_weight, runtime_scale=runtime_scale):
                return dynamic_w8a8_linear(x, runtime_weight, runtime_scale)

            expected, actual = original().cpu().float(), candidate().cpu().float()
            if not torch.isfinite(actual).all():
                raise RuntimeError("nonfinite whole-K INT8 output")
            native_channel_timing = None
            if resource:
                qa, sx = torch_npu.npu_dynamic_quant(x)
                sx = sx.unsqueeze(-1).expand(-1, 8).contiguous()
                channel_ends = torch.tensor([rows], dtype=torch.int64, device=x.device)
                integer_dot = qa.cpu().int() @ quantized.int().t()
                channel_reference = (integer_dot.float() * channel_scale.unsqueeze(0) * sx.cpu()[:, :1]).half()
                channel_actual = resource(qa, channel_bank, channel_scales, sx, channel_ends).cpu()
                torch.testing.assert_close(channel_actual, channel_reference, rtol=0, atol=0)

                def channel_run(
                    x=x, channel_bank=channel_bank, channel_scales=channel_scales, channel_ends=channel_ends
                ):
                    qa, sx = torch_npu.npu_dynamic_quant(x)
                    sx = sx.unsqueeze(-1).expand(-1, 8).contiguous()
                    return resource(qa, channel_bank, channel_scales, sx, channel_ends)

                native_channel_timing = timed(channel_run, args.iterations, args.repeats)
            error = actual - expected
            order = (original, candidate) if rows % 2 else (candidate, original)
            timing = {
                ("native" if f is original else "whole_k_int8"): timed(f, args.iterations, args.repeats) for f in order
            }
            records.append(
                {
                    "projection": projection,
                    "rows": rows,
                    "weight_relative_l2": weight_error,
                    "output_relative_l2": float(error.norm() / expected.norm().clamp_min(1e-12)),
                    "output_max_abs": float(error.abs().max()),
                    "exact": torch.equal(actual, expected),
                    "timing": timing,
                    "native_channel_timing": native_channel_timing,
                    "native_channel_exact_integer_recipe": resource is not None,
                }
            )
    after = server_activity(args.base_url)
    if after[0] or after[1] != before[1]:
        raise RuntimeError("traffic overlapped; discard math timings")
    record = {
        "scope": "single real expert, synthetic activations, no routing/shared/collectives",
        "serving_admitted": False,
        "quality_equivalent": False,
        "server_modified": False,
        "layer": args.layer,
        "expert": args.expert,
        "device": args.device,
        "cases": records,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
