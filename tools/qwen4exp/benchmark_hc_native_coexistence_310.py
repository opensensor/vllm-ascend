# SPDX-License-Identifier: Apache-2.0
"""Deferred isolated gate: keep residual graphs alive while loading HC mean."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.benchmark_native_hc_mean_310 import reference as mean_reference
from tools.qwen4exp.benchmark_native_hc_residual_310 import PATTERNS, changed_inputs
from tools.qwen4exp.benchmark_native_hc_residual_310 import reference as residual_reference

WIDTH = 2560
TOKEN_COUNTS = (3, 6, 640, 2560)
QUEUED_GROUP_SIZE = 4


def check_pair(residual, mean, hyper, block, injection, gate):
    expected_residual = residual_reference(hyper, block, injection)
    actual_residual = residual(hyper, block, injection)
    expected_mean = mean_reference(expected_residual.float(), gate)
    actual_mean = mean(actual_residual.float(), gate)
    checks = compare_outputs((expected_residual, expected_mean), (actual_residual, actual_mean))
    if not gate_passed(checks):
        raise AssertionError(f"native HC composition failed: {checks}")
    return checks


def compose_pair(residual, mean, hyper, block, injection, gate):
    output = residual(hyper, block, injection)
    return output, mean(output.float(), gate)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("residual-library", "residual-binary", "mean-library", "mean-binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1:
        parser.error("use a new output path and positive iteration count")
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture
    from tools.qwen4exp.direct_hc_mean import DirectHCMean
    from tools.qwen4exp.direct_hc_residual import DirectHCResidual

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    report = {"scope": "isolated_native_coexistence_not_model_performance", "passed": False, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(value):
        report["records"].append(value)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    generator = torch.Generator().manual_seed(8127)
    inputs = {}
    old_graphs = {}
    try:
        torch.ops.load_library(str(args.residual_library))
        residual = DirectHCResidual(str(args.residual_binary))
        for tokens in TOKEN_COUNTS:
            hyper = torch.empty(tokens, 4 * WIDTH, dtype=torch.float16, device="npu")
            block = torch.empty(tokens, WIDTH, dtype=torch.float16, device="npu")
            injection = torch.empty(tokens, 4, dtype=torch.float32, device="npu")
            gate = torch.full_like(hyper, 0.5)
            inputs[tokens] = hyper, block, injection, gate
            changed_inputs(hyper, block, injection, "random", generator)
            actual = residual(hyper, block, injection)
            checks = compare_outputs(residual_reference(hyper, block, injection), actual)
            assert gate_passed(checks), checks
            record({"phase": "residual_before_mean_load", "tokens": tokens, "checks": checks})
            if tokens <= 6:
                old_graphs[tokens] = capture(lambda tensors=inputs[tokens]: residual(*tensors[:3]))

        torch.npu.synchronize()
        torch.ops.load_library(str(args.mean_library))
        mean = DirectHCMean(str(args.mean_binary))
        composed_graphs = {
            tokens: capture(lambda tensors=tensors: compose_pair(residual, mean, *tensors))
            for tokens, tensors in inputs.items()
            if tokens <= 6
        }
        for tokens, tensors in inputs.items():
            hyper, block, injection, gate = tensors
            for pattern in PATTERNS:
                changed_inputs(hyper, block, injection, pattern, generator)
                checks = check_pair(residual, mean, *tensors)
                record({"phase": "both_libraries_loaded", "tokens": tokens, "pattern": pattern, "checks": checks})
                if tokens in old_graphs:
                    graph, output = old_graphs[tokens]
                    graph.replay()
                    torch.npu.synchronize()
                    checks = compare_outputs(residual_reference(hyper, block, injection), output)
                    assert gate_passed(checks), checks
                    record(
                        {
                            "phase": "old_residual_graph_after_mean_load",
                            "tokens": tokens,
                            "pattern": pattern,
                            "checks": checks,
                        }
                    )
                    graph, outputs = composed_graphs[tokens]
                    graph.replay()
                    torch.npu.synchronize()
                    expected = residual_reference(hyper, block, injection)
                    checks = compare_outputs((expected, mean_reference(expected.float(), gate)), outputs)
                    assert gate_passed(checks), checks
                    record({"phase": "composed_graph", "tokens": tokens, "pattern": pattern, "checks": checks})

        # Change allocation sizes and queue both operators together repeatedly.
        # Synchronizations bound queued memory; this is a stability gate, not timing.
        for hyper, block, injection, _ in inputs.values():
            changed_inputs(hyper, block, injection, "random", generator)
        pending = []
        for iteration in range(args.iterations):
            tokens = TOKEN_COUNTS[iteration % len(TOKEN_COUNTS)]
            hyper, block, injection, gate = inputs[tokens]
            # Queue device-only updates and both kernels before any readback.
            block.add_(0.125)
            expected = residual_reference(hyper, block, injection)
            actual = residual(hyper, block, injection)
            expected_mean = mean_reference(expected.float(), gate)
            actual_mean = mean(actual.float(), gate)
            pending.append((iteration, tokens, expected, expected_mean, actual, actual_mean))
            if len(pending) == QUEUED_GROUP_SIZE or iteration == args.iterations - 1:
                torch.npu.synchronize()
                for index, rows, expected, expected_mean, actual, actual_mean in pending:
                    checks = compare_outputs((expected, expected_mean), (actual, actual_mean))
                    assert gate_passed(checks), checks
                    record({"phase": "alternating_shapes", "iteration": index, "tokens": rows, "checks": checks})
                pending.clear()
        torch.npu.synchronize()
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
