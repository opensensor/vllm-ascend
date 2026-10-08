# SPDX-License-Identifier: Apache-2.0
"""Isolate mHC post-mixing scratch and arithmetic before changing serving.

The streaming candidates bound live intermediates to one output-sized term.
They are experimental: changed reduction/FMA order needs a serving quality gate.
This tool never changes model dispatch or starts a server.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def reference_post(x, residual, post_mix, comb_mix):
    mixed = torch.einsum("...ij,...ih->...jh", comb_mix.float(), residual.float())
    term = post_mix.float() * x.unsqueeze(-2).float()
    return (mixed + term).to(residual.dtype)


def streaming_post(x, residual, post_mix, comb_mix, *, fused_add=False):
    streams = residual.shape[-2]
    if streams < 1:
        raise ValueError("at least one residual stream is required")
    residual_fp32 = residual.float()
    comb_fp32 = comb_mix.float()
    output = residual_fp32[..., :1, :] * comb_fp32[..., 0, :, None]
    for stream in range(1, streams):
        values = residual_fp32[..., stream : stream + 1, :]
        weights = comb_fp32[..., stream, :, None]
        if fused_add:
            output.addcmul_(values, weights)
        else:
            output.add_(values * weights)
    output.add_(post_mix.float() * x.unsqueeze(-2).float())
    return output.to(residual.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 640, 1024, 1280])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--input-state", choices=["fp16_rounded", "fp32"], default="fp16_rounded")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or any(n < 1 for n in args.tokens):
        parser.error("output must be new; tokens and repeats must be positive")
    # Device import is intentionally confined to the executable benchmark.
    import torch_npu

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    operations = {
        "einsum": reference_post,
        "streaming": streaming_post,
        "addcmul": lambda *values: streaming_post(*values, fused_add=True),
    }
    cases = []
    for tokens in args.tokens:
        torch.manual_seed(3102026 + tokens)
        inputs = (
            torch.randn(tokens, 4096).half().npu(),
            torch.randn(tokens, 4, 4096).npu(),
            torch.sigmoid(torch.randn(tokens, 4, 1)).npu(),
            torch.softmax(torch.randn(tokens, 4, 4), dim=-1).npu(),
        )
        if args.input_state == "fp16_rounded":
            # _round_mhc_state(..., use_fp16=True) carries FP16-rounded values
            # in FP32 tensors. Raw random FP32 states are a different contract.
            inputs = (inputs[0], *(value.half().float() for value in inputs[1:]))
        expected = reference_post(*inputs).cpu()
        for name, operation in operations.items():
            for _ in range(3):
                output = operation(*inputs)
            actual = output.cpu()
            del output
            torch.npu.synchronize()
            torch.npu.empty_cache()
            before = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            output = operation(*inputs)
            torch.npu.synchronize()
            peak_delta = torch.npu.max_memory_allocated() - before
            del output
            cases.append(
                {
                    "tokens": tokens,
                    "operation": name,
                    "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                    "fp16_round_mismatches": int((actual.half() != expected.half()).sum()),
                    "finite": bool(torch.isfinite(actual).all()),
                    "peak_allocated_delta_bytes": peak_delta,
                    "samples_ms": [],
                }
            )
        # Rotate order to reduce systematic warmup/temperature bias.
        names = list(operations)
        for repeat in range(args.repeats):
            for index in range(len(names)):
                name = names[(index + repeat) % len(names)]
                torch.npu.synchronize()
                start = time.perf_counter()
                output = operations[name](*inputs)
                torch.npu.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                cases[-len(names) + names.index(name)]["samples_ms"].append(elapsed)
                del output
        del inputs
    for case in cases:
        case["median_ms"] = statistics.median(case["samples_ms"])
    result = {"input_state": args.input_state, "cases": cases, "serving_tested": False, "graph_tested": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
