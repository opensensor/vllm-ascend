# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen4Exp grouped RMSNorm with native 310P RMSNorm variants.

This is a diagnostic benchmark, not a serving-path change. It uses the
2048-token, four-stream activation geometry observed in real prefill traces.
"""

import argparse
import json
import time

import torch
import torch_npu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if min(args.tokens, args.streams, args.hidden, args.repeats) < 1:
        parser.error("all numeric arguments must be positive")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")

    torch.manual_seed(310)
    device = "npu:0"
    epsilon = 1e-6
    x = torch.randn(args.tokens, args.streams * args.hidden, device=device, dtype=torch.float16)
    weight = torch.randn(args.streams * args.hidden, device=device, dtype=torch.float16) * 0.1
    affine_fp32 = 1.0 + weight.float().view(args.streams, args.hidden)
    unit_fp32 = torch.ones(args.hidden, device=device, dtype=torch.float32)
    unit_fp16 = unit_fp32.half()

    def eager() -> torch.Tensor:
        grouped = x.float().view(args.tokens, args.streams, args.hidden)
        variance = grouped.square().mean(dim=-1, keepdim=True)
        return (grouped * torch.rsqrt(variance + epsilon) * (1.0 + weight.float()).view(1, args.streams, -1)).view(
            args.tokens, -1
        )

    def native_fp32() -> torch.Tensor:
        normalized, _ = torch_npu.npu_rms_norm(x.float().view(-1, args.hidden), unit_fp32, epsilon)
        return (normalized.view(args.tokens, args.streams, args.hidden) * affine_fp32).view(args.tokens, -1)

    def native_fp16() -> torch.Tensor:
        normalized, _ = torch_npu.npu_rms_norm(x.view(-1, args.hidden), unit_fp16, epsilon)
        return (normalized.float().view(args.tokens, args.streams, args.hidden) * affine_fp32).view(args.tokens, -1)

    def measure(fn):
        result = fn()
        torch_npu.npu.synchronize()
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = fn()
            torch_npu.npu.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
        return result, durations

    reference, eager_ms = measure(eager)
    results = {"eager_ms": eager_ms}
    for name, fn in (("native_fp32", native_fp32), ("native_fp16", native_fp16)):
        try:
            output, durations = measure(fn)
            error = output.float() - reference.float()
            results[f"{name}_ms"] = durations
            results[f"{name}_max_abs_difference"] = error.abs().max().item()
            results[f"{name}_relative_rms_difference"] = (
                error.square().mean().sqrt() / reference.float().square().mean().sqrt()
            ).item()
        except RuntimeError as exc:
            results[f"{name}_error"] = str(exc)
    print(json.dumps({"tokens": args.tokens, "streams": args.streams, "hidden": args.hidden, **results}))


if __name__ == "__main__":
    main()
