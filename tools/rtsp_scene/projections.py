"""Benchmark real dense Qwen3-VL projections; operator gains are not service gains."""

import argparse
import json
import statistics
import time
from pathlib import Path

from tools.rtsp_scene.quantize import symmetric_int8


def timed(function, synchronize, iterations: int, repeats: int) -> dict:
    for _ in range(3):
        function()
    synchronize()
    trials = []
    for _ in range(repeats):
        before = time.perf_counter()
        for _ in range(iterations):
            function()
        synchronize()
        trials.append((time.perf_counter() - before) * 1000 / iterations)
    return {"median_ms": statistics.median(trials), "trials_ms": trials}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 16, 128])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min([*args.tokens, args.iterations, args.repeats]) <= 0:
        parser.error("Use a new output file and positive counts")
    # Hardware dependencies are restricted to this isolated benchmark worker.
    import torch
    import torch_npu
    from safetensors import safe_open

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20261012)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.0."
    groups = {
        "qkv": [prefix + "self_attn." + item + ".weight" for item in ("q_proj", "k_proj", "v_proj")],
        "output": [prefix + "self_attn.o_proj.weight"],
        "gate_up": [prefix + "mlp." + item + ".weight" for item in ("gate_proj", "up_proj")],
        "down": [prefix + "mlp.down_proj.weight"],
        "head": ["lm_head.weight"],
    }
    report = {"scope": "real weights, synthetic activations; isolated operator timings", "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for projection, names in groups.items():
        pieces = []
        for name in names:
            with safe_open(args.model / index[name], framework="pt", device="cpu") as shard:
                pieces.append(shard.get_tensor(name).half())
        weight_cpu = torch.cat(pieces, dim=0)
        qweight, scale, _ = symmetric_int8(weight_cpu)
        weight = torch_npu.npu_format_cast(weight_cpu.npu(), 29)
        quant_weight = torch_npu.npu_format_cast(qweight.npu(), 29).transpose(0, 1)
        quant_scale = scale.flatten().npu()
        for tokens in args.tokens:
            x = torch.randn(tokens, weight.shape[1], dtype=torch.float16, device="npu") * 0.1

            def fp16(x=x, weight=weight):
                return torch.nn.functional.linear(x, weight)

            def int8(x=x, quant_weight=quant_weight, quant_scale=quant_scale):
                qx, activation_scale = torch_npu.npu_dynamic_quant(x)
                return torch_npu.npu_quant_matmul(
                    qx, quant_weight, quant_scale, pertoken_scale=activation_scale, output_dtype=torch.float16
                )

            expected, actual = fp16().float(), int8().float()
            relative_error = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
            record = {
                "projection": projection,
                "tokens": tokens,
                "weight_shape": list(weight_cpu.shape),
                "relative_l2_error": relative_error,
                "finite": bool(torch.isfinite(actual).all()),
                "fp16": timed(fp16, torch.npu.synchronize, args.iterations, args.repeats),
                "int8_dynamic": timed(int8, torch.npu.synchronize, args.iterations, args.repeats),
            }
            record["speedup"] = record["fp16"]["median_ms"] / record["int8_dynamic"]["median_ms"]
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        del weight, quant_weight, quant_scale, pieces, weight_cpu, qweight, scale, x, expected, actual
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
