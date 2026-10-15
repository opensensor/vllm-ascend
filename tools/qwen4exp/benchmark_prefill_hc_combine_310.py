# SPDX-License-Identifier: Apache-2.0
"""Strict real-weight gates for the prefill-only residual addcmul candidate."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        choices=(
            "prefill_addcmul",
            "padded_injection",
            "padded_injection_and_prefill",
            "padded_reciprocal",
            "native_residual",
            "native_mean",
        ),
        default="prefill_addcmul",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    if args.candidate in ("native_residual", "native_mean") and (args.library is None or args.binary is None):
        parser.error("native qualification requires library and binary paths")
    if args.output.exists():
        parser.error("preserve existing evidence")
    # Isolated hardware process; real model imports stay out of host tooling.
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing
    from tools.qwen4exp.direct_hc_mean import DirectHCMean
    from tools.qwen4exp.direct_hc_residual import DirectHCResidual
    from tools.qwen4exp.hc_checkpoint import load_hyperconnection
    from tools.qwen4exp.resident_candidates import hc_reciprocal_scaling as reciprocal
    from tools.qwen4exp.resident_candidates.native_hc_mean import make_mix
    from tools.qwen4exp.resident_candidates.native_hc_residual import make_combine
    from tools.qwen4exp.resident_candidates.padded_hc_injection import combine as padded_combine
    from tools.qwen4exp.resident_candidates.padded_hc_injection import combine_with_prefill_addcmul
    from tools.qwen4exp.resident_candidates.prefill_hc_addcmul import combine as prefill_combine
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    native_combine = None
    native_mix = None
    if args.candidate == "native_residual":
        torch.npu.synchronize()
        torch.ops.load_library(str(args.library))
        native_combine = make_combine(DirectHCResidual(str(args.binary)))
    elif args.candidate == "native_mean":
        torch.npu.synchronize()
        torch.ops.load_library(str(args.library))
        native_mix = make_mix(DirectHCMean(str(args.binary)))
    combine = {
        "prefill_addcmul": prefill_combine,
        "padded_injection": padded_combine,
        "padded_injection_and_prefill": combine_with_prefill_addcmul,
        "padded_reciprocal": reciprocal.combine,
        "native_residual": native_combine,
        "native_mean": None,
    }[args.candidate]
    report = {"scope": "real_hc_combine_synthetic_changed_activations_not_model_quality", "records": [], "passed": True}
    report["candidate"] = args.candidate
    report["baseline_candidate"] = (
        "padded_injection_and_prefill"
        if args.candidate in ("padded_reciprocal", "native_residual", "native_mean")
        else "original"
    )

    def evaluate(module, hyper, block, residuals, use_candidate):
        if args.candidate == "native_mean":
            mixed, saved = native_mix(module, hyper) if use_candidate else module.mix(hyper)
            return mixed, combine_with_prefill_addcmul(module, block, saved)
        if args.candidate == "padded_reciprocal":
            # Check and time the entire mix/combine pair against the selected
            # padded candidate, isolating reciprocal scaling's incremental cost.
            if use_candidate:
                mixed, saved = reciprocal.mix(module, hyper)
                return mixed, reciprocal.combine(module, block, saved)
            mixed, saved = module.mix(hyper)
            return mixed, combine_with_prefill_addcmul(module, block, saved)
        if args.candidate == "native_residual" and not use_candidate:
            return combine_with_prefill_addcmul(module, block, residuals)
        return combine(module, block, residuals) if use_candidate else module.combine(block, residuals)

    prefixes = (
        "model.language_model.layers.0.attn_hyper_connection",
        "model.language_model.layers.0.mlp_hyper_connection",
        "model.language_model.layers.47.mlp_hyper_connection",
        "mtp.layers.0.attn_hyper_connection",
    )
    for prefix in prefixes:
        module, keys = load_hyperconnection(args.checkpoint, prefix)
        module.npu()
        for parameter in module.parameters():
            if parameter.ndim == 2 and min(parameter.shape) >= 16:
                parameter.data = torch_npu.npu_format_cast(parameter.data, ACL_FORMAT_FRACTAL_NZ)
        module.prepare_norm_affine()
        for tokens in (3, 6, 2560):
            generator = torch.Generator().manual_seed(1165)
            gates = []
            for scale in (0.1, 0.0, 4.0):
                hyper = (torch.randn(tokens, module.hyper_hidden, generator=generator) * scale).half().npu()
                block = (torch.randn(tokens, module.hidden_size, generator=generator) * scale).half().npu()
                normalized = module._normalize(hyper)
                residuals = (hyper, normalized)
                expected = evaluate(module, hyper, block, residuals, False)
                actual = evaluate(module, hyper, block, residuals, True)
                gates.append(compare_outputs(expected, actual))
            record = {
                "prefix": prefix,
                "keys": keys,
                "tokens": tokens,
                "gates": gates,
                "passed": all(gate_passed(g) for g in gates),
            }
            if record["passed"]:
                functions = {
                    "baseline": lambda module=module, hyper=hyper, block=block, residuals=residuals: evaluate(
                        module, hyper, block, residuals, False
                    ),
                    "candidate": lambda module=module, hyper=hyper, block=block, residuals=residuals: evaluate(
                        module, hyper, block, residuals, True
                    ),
                }
                record["eager_timing"] = paired_timing(functions, 5, 7)
                if tokens <= 6:
                    captured = {name: capture(function) for name, function in functions.items()}
                    record["graph_gates"] = []
                    for scale in (0.1, 0.0, 4.0):
                        hyper.copy_((torch.randn(hyper.shape, generator=generator) * scale).half().npu())
                        block.copy_((torch.randn(block.shape, generator=generator) * scale).half().npu())
                        normalized.copy_(module._normalize(hyper))
                        for graph, _ in captured.values():
                            graph.replay()
                        torch.npu.synchronize()
                        expected = functions["baseline"]()
                        checks = {name: compare_outputs(expected, output) for name, (_, output) in captured.items()}
                        record["graph_gates"].append(checks)
                    record["passed"] &= all(
                        gate_passed(gate) for checks in record["graph_gates"] for gate in checks.values()
                    )
                    if record["passed"]:
                        record["graph_timing"] = paired_timing(
                            {name: graph.replay for name, (graph, _) in captured.items()}, 100, 7
                        )
                    del captured, graph
            report["passed"] &= record["passed"]
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"prefix": prefix, "tokens": tokens, "passed": record["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
