# SPDX-License-Identifier: Apache-2.0
"""Private real-weight DSpark projection probe; never mutates the loaded model."""

import gc
import statistics
import time
from pathlib import Path


def probe(worker):
    import torch
    import torch.nn.functional as F
    import torch_npu
    from safetensors import safe_open

    from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import dynamic_w8a8_linear, quantize_linear_weight

    if worker.rank != 0:
        return {"rank": worker.rank, "skipped": True}
    root = Path("/srv/ai/storage/experiments/qfn-dspark-20261019")
    device = worker.model_runner.device
    generator = torch.Generator().manual_seed(93217)
    report = {"rank": worker.rank, "scope": "actual weights, synthetic activations, eager projection only", "cases": []}

    def timing(fn):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(20):
                fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - started) * 1000 / 20)
        return statistics.median(samples)

    def measure(name, weight):
        # CPU quantization avoids transient FP32 copies in the resident NPU pool.
        weight = weight.to(torch.float16).contiguous()
        quantized, scales = quantize_linear_weight(weight)
        fp16 = torch_npu.npu_format_cast(weight.to(device), 29)
        int8 = torch_npu.npu_format_cast(quantized.to(device), 29).transpose(0, 1)
        scale_device = scales.to(device)
        for rows in (2, 7, 8, 28):
            cpu_x = torch.randn(rows, weight.shape[1], generator=generator, dtype=torch.float32).half()
            x = cpu_x.to(device)
            expected = F.linear(x, fp16)
            actual = dynamic_w8a8_linear(x, int8, scale_device)
            torch.npu.synchronize()
            # Validate matmul against the actual device quantizer's integers.
            # Its rounding/scales need not match the CPU helper's quantizer.
            device_qx, device_scale = torch_npu.npu_dynamic_quant(x)
            reference = (
                (device_qx.cpu().float() @ quantized.t().float()) * device_scale.cpu().float().reshape(-1, 1) * scales
            ).half()
            actual_cpu = actual.cpu().float()
            expected_cpu = expected.cpu().float()
            quant_error = float((actual_cpu - reference.float()).abs().max())
            # Kernel/reference discrepancy is separate from intentional INT8 error.
            if not torch.isfinite(actual_cpu).all():
                raise RuntimeError(f"nonfinite INT8 output in {name}")
            torch.testing.assert_close(actual_cpu, reference.float(), rtol=0.005, atol=0.02)
            fp16_ms = timing(lambda x=x, w=fp16: F.linear(x, w))
            int8_ms = timing(lambda x=x, w=int8, s=scale_device: dynamic_w8a8_linear(x, w, s))
            report["cases"].append(
                dict(
                    name=name,
                    weight_shape=list(weight.shape),
                    rows=rows,
                    fp16_ms=fp16_ms,
                    int8_ms=int8_ms,
                    speedup=fp16_ms / int8_ms,
                    relative_rms=float(
                        (actual_cpu - expected_cpu).square().mean().sqrt()
                        / expected_cpu.square().mean().sqrt().clamp_min(1e-12)
                    ),
                    quant_reference_max_error=quant_error,
                )
            )
        del fp16, int8, scale_device
        gc.collect()
        torch.npu.empty_cache()

    with torch.inference_mode(), safe_open(root / "draft/model.safetensors", framework="pt", device="cpu") as f:
        selections = {
            "fc": ["fc.weight"],
            "qkv": [f"layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v")],
            "o": ["layers.0.self_attn.o_proj.weight"],
            "gate_up": [f"layers.0.mlp.{p}_proj.weight" for p in ("gate", "up")],
            "down": ["layers.0.mlp.down_proj.weight"],
            "markov": ["markov_head.markov_w2.weight"],
        }
        for name, keys in selections.items():
            measure(name, torch.cat([f.get_tensor(k) for k in keys]))
        target = Path("/srv/ai/storage/models/Qwen3.8-Flash-Next-W4A16-G128-300i/float-00000.safetensors")
        with safe_open(target, framework="pt", device="cpu") as target_file:
            measure("lm_head_shard", target_file.get_slice("lm_head.weight")[:41408])
    report["passed"] = True
    return report
