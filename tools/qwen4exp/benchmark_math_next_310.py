# SPDX-License-Identifier: Apache-2.0
"""Isolated stock-op screening for remaining Qwen math opportunities."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.math_next_cases import CASE_NAMES, build_case
from tools.qwen4exp.math_probe_cases import PATTERNS, changed_inputs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[3, 6, 640, 2560])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or any(n <= 0 for n in args.tokens):
        parser.error("positive token counts and a fresh output path required")
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    report = {"scope": "stock_math_screening_not_model_quality_or_tok_s", "case": args.case, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        case = build_case(args.case, tokens, "npu", torch_npu)
        record = {"tokens": tokens, "description": case.description, "gates": []}
        functions = {
            "baseline": lambda case=case: case.baseline(*case.inputs),
            "candidate": lambda case=case: case.candidate(*case.inputs),
        }
        try:
            for index, pattern in enumerate(PATTERNS):
                values = changed_inputs(case.inputs, pattern, 6130 + index)
                for target, value in zip(case.inputs, values):
                    target.copy_(value)
                checks = compare_outputs(functions["baseline"](), functions["candidate"]())
                record["gates"].append({"pattern": pattern, "checks": checks})
            record["exact"] = all(gate_passed(gate["checks"]) for gate in record["gates"])
            # Record exploratory timing even for precision-changing stock operators.
            record["timing_qualified_for_serving"] = record["exact"]
            for target, value in zip(case.inputs, changed_inputs(case.inputs, "random", 8112)):
                target.copy_(value)
            record["eager_timing"] = paired_timing(functions, 5 if tokens >= 640 else 20, 3)
            if tokens <= 6 and record["exact"]:
                graphs = {name: capture(function) for name, function in functions.items()}
                record["graph_checks"] = []
                for target, value in zip(case.inputs, changed_inputs(case.inputs, "random", 8113)):
                    target.copy_(value)
                for graph, _ in graphs.values():
                    graph.replay()
                torch.npu.synchronize()
                expected = functions["baseline"]()
                for name, (_, output) in graphs.items():
                    record["graph_checks"].append({"arm": name, "checks": compare_outputs(expected, output)})
                record["graph_exact"] = all(gate_passed(g["checks"]) for g in record["graph_checks"])
                if record["graph_exact"]:
                    record["graph_timing"] = paired_timing(
                        {name: graph.replay for name, (graph, _) in graphs.items()}, 50, 3
                    )
                del graphs, graph
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["exact"] = False
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise
        report["records"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": args.case, "tokens": tokens, "exact": record["exact"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
