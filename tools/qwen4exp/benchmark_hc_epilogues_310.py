# SPDX-License-Identifier: Apache-2.0
"""Isolated equivalence and paired timing for HC conversion/activation epilogues."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing
from tools.qwen4exp.hc_epilogues import STREAMS, WIDTH, HCEpilogues, make_mix


def exact(left, right):
    return torch.equal(left.cpu().view(torch.uint8), right.cpu().view(torch.uint8))


@torch.no_grad()
def main():
    """Gate exceptional values first, then compare real-weight HC calls and timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(args.bridge))
    kernels = [
        torch.classes.qwen_hc_combine_v1.Kernel(str(args.binary), name)
        for name in ("qwen_hc_affine_v1", "qwen_hc_mean_epilogue_v1")
    ]
    owner = HCEpilogues(*kernels, torch.ops.qwen_hc_combine_v1.launch)
    report = dict(scope="isolated operators and optional real-weight HC, not model serving", passed=True, records=[])

    def save(record):
        report["records"].append(record)
        report["passed"] &= record["passed"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    generator = torch.Generator().manual_seed(15891)
    for rows in (1, 3, 128, 259, 515, 2048):
        source = torch.randn(rows, STREAMS * WIDTH, generator=generator).npu()
        weight = torch.randn(STREAMS * WIDTH, generator=generator).npu()
        gate = torch.sigmoid(torch.randn(rows, STREAMS * WIDTH, generator=generator)).half().npu()
        for pattern in ("random", "exceptional", "signed_zero", "subnormal", "saturation"):
            if pattern == "exceptional":
                source[:, :8] = torch.tensor(
                    [0.0, -0.0, 2**-24, -(2**-24), float("inf"), -float("inf"), float("nan"), 1e20],
                    device=source.device,
                )
            elif pattern in ("signed_zero", "subnormal", "saturation"):
                source.fill_({"signed_zero": -0.0, "subnormal": 2**-24, "saturation": 1e20}[pattern])
                weight.fill_(1.0)
                gate.fill_(1.0)
            expected = source * weight
            full, narrow = owner.affine(source, weight)
            mean = owner.mean(source, gate)
            expected_mean = (source.view(rows, STREAMS, WIDTH) * gate.view(rows, STREAMS, WIDTH)).mean(1).half()
            checks = dict(
                full=exact(expected, full), narrow=exact(expected.half(), narrow), mean=exact(expected_mean, mean)
            )
            save(
                dict(
                    rows=rows,
                    pattern=pattern,
                    checks=checks,
                    passed=all(checks.values()),
                    mean_max_error=(expected_mean.float() - mean.float()).abs().max().item(),
                    mean_mismatches=int((expected_mean != mean).sum().item()),
                )
            )
        if not report["passed"]:
            raise RuntimeError("private epilogue equivalence failed")
        source = torch.randn(rows, STREAMS * WIDTH, generator=generator).npu()
        for operation in ("affine", "mean"):
            if operation == "affine":

                def baseline(source=source, weight=weight):
                    product = source * weight
                    return product, product.half()

                candidate = lambda source=source, weight=weight: owner.affine(source, weight)
            else:
                baseline = lambda source=source, gate=gate, rows=rows: (
                    (source.view(rows, STREAMS, WIDTH) * gate.view(rows, STREAMS, WIDTH)).mean(1).half()
                )
                candidate = lambda source=source, gate=gate: owner.mean(source, gate)
            for _ in range(3):
                baseline()
                candidate()
            save(
                dict(
                    rows=rows,
                    operation=operation,
                    passed=True,
                    timing=paired_timing(dict(baseline=baseline, candidate=candidate), 5, 5),
                )
            )
    if args.checkpoint:
        from tools.qwen4exp.hc_checkpoint import load_hyperconnection

        for prefix in (
            "model.language_model.layers.0.attn_hyper_connection",
            "model.language_model.layers.0.mlp_hyper_connection",
            "model.language_model.layers.47.mlp_hyper_connection",
            "mtp.layers.0.attn_hyper_connection",
        ):
            module, keys = load_hyperconnection(args.checkpoint, prefix)
            module = module.npu()
            for parameter in module.parameters():
                if parameter.ndim == 2 and min(parameter.shape) >= 16:
                    parameter.data = torch_npu.npu_format_cast(parameter.data, 29)
            module.prepare_norm_affine()
            candidates = {"baseline": type(module).mix}
            candidates.update(
                {
                    "affine": make_mix(owner, type(module).mix, mean=False),
                    "mean": make_mix(owner, type(module).mix, affine=False),
                    "both": make_mix(owner, type(module).mix),
                }
            )
            for rows in (128, 515, 2048):
                values = torch.randn(rows, STREAMS * WIDTH, generator=generator).half().npu()
                reference = module.mix(values)
                for name, function in candidates.items():
                    actual = function(module, values)
                    save(
                        dict(
                            prefix=prefix,
                            rows=rows,
                            mode=name,
                            keys=keys,
                            passed=exact(reference[0], actual[0]) and exact(reference[1][1], actual[1][1]),
                        )
                    )
                if not report["passed"]:
                    raise RuntimeError("real-weight HC equivalence failed")
                functions = {
                    name: lambda function=function, module=module, values=values: function(module, values)
                    for name, function in candidates.items()
                }
                save(dict(prefix=prefix, rows=rows, passed=True, timing=paired_timing(functions, 3, 7)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
