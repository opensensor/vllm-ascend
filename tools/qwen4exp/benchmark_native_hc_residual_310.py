# SPDX-License-Identifier: Apache-2.0
"""Strict isolated gates for the runtime-compiled HC residual vector kernel."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed

PATTERNS = ("random", "zero", "large", "subnormal", "signed_zero", "nonfinite")


def reference(hyper, block, injection):
    rows, width = block.shape
    residual = hyper.view(rows, 4, width).float()
    if rows >= 2560:
        output = residual.addcmul_(block.float()[:, None], injection[:, :, None])
    else:
        output = residual + block.float()[:, None] * injection[:, :, None]
    return output.flatten(1).half()


def changed_inputs(hyper, block, injection, pattern, generator):
    if pattern == "random":
        for tensor in (hyper, block):
            tensor.copy_(torch.randn(tensor.shape, generator=generator).half().to(tensor.device))
        injection.copy_((2 * torch.rand(injection.shape, generator=generator)).half().to(injection.device))
    elif pattern == "zero":
        hyper.zero_()
        block.zero_()
        injection.fill_(1)
    elif pattern == "large":
        hyper.fill_(60000)
        block.fill_(60000)
        injection.fill_(2)
        hyper[:, ::2].neg_()
        block[:, ::2].neg_()
    elif pattern == "subnormal":
        hyper.fill_(2**-24)
        block.fill_(-(2**-24))
        injection.fill_(0.5)
    elif pattern == "signed_zero":
        hyper.fill_(-0.0)
        block.fill_(-0.0)
        injection.fill_(0.0)
        injection[:, ::2].fill_(-0.0)
    elif pattern == "nonfinite":
        hyper.fill_(float("nan"))
        block.fill_(float("inf"))
        injection.fill_(1.0)
        hyper[:, ::3].fill_(-float("inf"))
    else:
        raise ValueError(pattern)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence")
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing
    from tools.qwen4exp.direct_hc_residual import DirectHCResidual

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    sentinel = torch.arange(16, device="npu", dtype=torch.float32)
    pointer = sentinel.data_ptr()
    torch.npu.synchronize()
    torch.ops.load_library(str(args.library))
    op = DirectHCResidual(str(args.binary))
    report = {"scope": "synthetic_residual_math_not_complete_HC_or_model_quality", "records": [], "passed": True}
    for rows, width in ((3, 512), (6, 512), (3, 2560), (6, 2560), (640, 2560), (2560, 2560)):
        hyper = torch.empty(rows, 4 * width, dtype=torch.float16, device="npu")
        block = torch.empty(rows, width, dtype=torch.float16, device="npu")
        injection = torch.empty(rows, 4, dtype=torch.float16, device="npu")
        generator = torch.Generator().manual_seed(6512)
        functions = {
            "baseline": lambda hyper=hyper, block=block, injection=injection: reference(hyper, block, injection),
            "candidate": lambda hyper=hyper, block=block, injection=injection: op(hyper, block, injection),
        }
        record = {"tokens": rows, "width": width, "gates": [], "passed": True}
        for pattern in PATTERNS:
            changed_inputs(hyper, block, injection, pattern, generator)
            gates = compare_outputs(functions["baseline"](), functions["candidate"]())
            record["gates"].append({"pattern": pattern, "checks": gates})
            record["passed"] &= gate_passed(gates)
        if record["passed"]:
            changed_inputs(hyper, block, injection, "random", generator)
            record["eager_timing"] = paired_timing(functions, 5 if rows >= 640 else 20, 7)
            if rows <= 6:
                captured = {name: capture(function) for name, function in functions.items()}
                record["graph_gates"] = []
                for pattern in PATTERNS:
                    changed_inputs(hyper, block, injection, pattern, generator)
                    for graph, _ in captured.values():
                        graph.replay()
                    torch.npu.synchronize()
                    expected = functions["baseline"]()
                    gates = {name: compare_outputs(expected, output) for name, (_, output) in captured.items()}
                    record["graph_gates"].append({"pattern": pattern, "checks": gates})
                    record["passed"] &= all(gate_passed(check) for check in gates.values())
                if record["passed"]:
                    changed_inputs(hyper, block, injection, "random", generator)
                    record["graph_timing"] = paired_timing(
                        {name: graph.replay for name, (graph, _) in captured.items()}, 100, 7
                    )
                del captured, graph
        report["passed"] &= record["passed"]
        report["records"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"tokens": rows, "width": width, "passed": record["passed"]}), flush=True)
    torch.npu.synchronize()
    assert sentinel.data_ptr() == pointer
    torch.testing.assert_close(sentinel.cpu(), torch.arange(16, dtype=torch.float32), rtol=0, atol=0)
    report["sentinel_storage_preserved"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
