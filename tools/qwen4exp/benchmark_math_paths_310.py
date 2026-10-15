# SPDX-License-Identifier: Apache-2.0
"""Queueable small math comparisons; no server, HCCL or checkpoint loading."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools.qwen4exp.math_probe_cases import CASE_NAMES, PATTERNS, build_case, changed_inputs


def compare_outputs(expected, actual):
    expected = (expected,) if isinstance(expected, torch.Tensor) else expected
    actual = (actual,) if isinstance(actual, torch.Tensor) else actual
    if len(expected) != len(actual):
        raise ValueError("candidate output count differs")
    result = []
    for left, right in zip(expected, actual):
        left, right = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("candidate output shape/dtype differs")
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        raw = right.view(torch.uint8)
        result.append(
            {
                "dtype": str(right.dtype),
                "shape": list(right.shape),
                "finite": finite,
                "bitwise_equal": torch.equal(left.view(torch.uint8), raw),
                "fp16_round_equal": torch.equal(left.half(), right.half()) if right.is_floating_point() else None,
                "changed_elements": int(torch.count_nonzero(left != right)),
                "max_abs_error": float((left.double() - right.double()).abs().max()),
                "actual_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            }
        )
    return result


def gate_passed(results):
    return all(item["finite"] and item["bitwise_equal"] for item in results)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="host-only enumeration; no NPU imports")
    parser.add_argument("--case", choices=CASE_NAMES, nargs="+", default=list(CASE_NAMES))
    parser.add_argument("--tokens", type=int, nargs="+", default=[3, 6, 2560])
    parser.add_argument("--patterns", choices=PATTERNS, nargs="+", default=list(PATTERNS))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(tokens < 1 for tokens in args.tokens):
        parser.error("tokens and repeats must be positive")
    if args.list:
        print(json.dumps([{"case": name, "tokens": tokens} for name in args.case for tokens in args.tokens], indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("supply a new output path")
    # Isolation-only imports. Enumeration and contract tests never init NPU.
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    report = {
        "scope": "synthetic_basic_math_not_model_performance_or_quality",
        "timing_gate": "exact_finite_outputs_on_all_changed_inputs",
        "dynamic_casts_and_copies_included": True,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for name in args.case:
        for tokens in args.tokens:
            torch.npu.reset_peak_memory_stats()
            case = build_case(name, tokens, "npu", torch_npu)
            record = {
                "case": name,
                "tokens": tokens,
                "description": case.description,
                "input_storage_bytes": sum(t.untyped_storage().nbytes() for t in case.inputs),
                "eager_gates": [],
            }
            functions = {
                "baseline": lambda case=case: case.baseline(*case.inputs),
                "candidate": lambda case=case: case.candidate(*case.inputs),
            }
            graphs = {}
            try:
                for index, pattern in enumerate(args.patterns):
                    updates = changed_inputs(case.inputs, pattern, 1105 + index)
                    for target, source in zip(case.inputs, updates):
                        target.copy_(source)
                    result = compare_outputs(functions["baseline"](), functions["candidate"]())
                    record["eager_gates"].append({"pattern": pattern, "outputs": result})
                    del updates
                exact = all(gate_passed(item["outputs"]) for item in record["eager_gates"])
                record["exact_eager_passed"] = exact
                if exact:
                    iterations = 5 if tokens >= 2560 else 50
                    record["eager"] = paired_timing(functions, iterations, args.repeats)
                    # Large prefill graphs are unnecessary for this decode
                    # suite; they would consume memory beyond the tiny probe.
                    if tokens <= 128:
                        for arm, function in functions.items():
                            graphs[arm] = capture(function)
                        record["graph_gates"] = []
                        for index, pattern in enumerate(args.patterns):
                            updates = changed_inputs(case.inputs, pattern, 1205 + index)
                            for target, source in zip(case.inputs, updates):
                                target.copy_(source)
                            for graph, _ in graphs.values():
                                graph.replay()
                            torch.npu.synchronize()
                            expected = functions["baseline"]()
                            arm_results = {
                                arm: compare_outputs(expected, output) for arm, (_, output) in graphs.items()
                            }
                            record["graph_gates"].append({"pattern": pattern, "outputs": arm_results})
                            del updates
                        record["exact_graph_passed"] = all(
                            gate_passed(outputs)
                            for item in record["graph_gates"]
                            for outputs in item["outputs"].values()
                        )
                        if record["exact_graph_passed"]:
                            record["graph"] = paired_timing(
                                {arm: graph.replay for arm, (graph, _) in graphs.items()}, 100, args.repeats
                            )
                else:
                    record["timing_skipped"] = "exact parity failed; approximation needs separate quality qualification"
            except (RuntimeError, ValueError, AttributeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
                # A capture/device failure can poison the process; do not run
                # subsequent cases after such an exception. Narrow --case to
                # the failing arm in a fresh process for further diagnostics.
                report["records"].append(record)
                save()
                raise
            report["records"].append(record)
            # Combined tensor allocator peak for both arms, not per-arm memory
            # and not a device-wide CANN/driver workspace measurement.
            record["peak_tensor_allocated_bytes"] = torch.npu.max_memory_allocated()
            record["peak_tensor_reserved_bytes"] = torch.npu.max_memory_reserved()
            save()
            print(json.dumps({"case": name, "tokens": tokens, "exact": record["exact_eager_passed"]}), flush=True)
            graphs.clear()
            del case, functions
            torch.npu.synchronize()


if __name__ == "__main__":
    main()
