# SPDX-License-Identifier: Apache-2.0
"""Real HC projection probes using 310P's supported single-bank FP16 GMM."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.qwen4exp.benchmark_fp16_projection_paths_310 import DEFAULT_PREFIX, PROJECTIONS
from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed

CUBE_ALIGNMENT = 16
BATCH_MATMUL_ARMS = ("bmm_cached_bank", "matmul_broadcast_bank")


def select_graph_arms(qualified, skip_batchmatmul):
    """Retain baseline and GMM after the isolated aclop BatchMatMul failure."""
    return {
        name: function for name, function in qualified.items() if not skip_batchmatmul or name not in BATCH_MATMUL_ARMS
    }


def projection_functions(inputs, weight, cached_bank, group_ends, grouped_matmul, padded_weight=None):
    """Only constant bank/ends preparation is excluded; input casts are timed."""
    columns = weight.shape[0]
    functions = {
        "linear_nz_weight": lambda: F.linear(inputs.to(weight.dtype), weight),
        "gmm_cached_bank": lambda: grouped_matmul(
            x=[inputs.to(weight.dtype)],
            weight=[cached_bank],
            group_list=group_ends,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )[0][:, :columns],
        "bmm_cached_bank": lambda: torch.bmm(inputs.to(weight.dtype).unsqueeze(0), cached_bank)[0, :, :columns],
        "matmul_broadcast_bank": lambda: torch.matmul(inputs.to(weight.dtype), cached_bank)[0, :, :columns],
    }
    if padded_weight is not None:
        functions["linear_padded_nz_weight"] = lambda: F.linear(inputs.to(weight.dtype), padded_weight)[:, :columns]
    if columns % CUBE_ALIGNMENT == 0:
        # No static weight copy: preserve the original NZ storage with a 3D
        # transposed view. Whether CANN can consume this view is a probe gate.
        weight_view = weight.unsqueeze(0).transpose(1, 2)
        functions["gmm_original_weight_view"] = lambda: grouped_matmul(
            x=[inputs.to(weight.dtype)],
            weight=[weight_view],
            group_list=group_ends,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )[0]
    return functions


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--module-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--projection", choices=PROJECTIONS, required=True)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--arm", nargs="+", help="optionally isolate unsupported arms in fresh processes")
    parser.add_argument(
        "--skip-batchmatmul-graphs", action="store_true", help="retain BMM eager gates but omit known aclop graph paths"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokens < 1 or args.repeats < 1:
        parser.error("use a new output and positive tokens/repeats")
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing
    from tools.qwen4exp.hc_checkpoint import load_hyperconnection
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    module, keys = load_hyperconnection(args.checkpoint, args.module_prefix)
    cpu_weight = getattr(module, args.projection).detach()
    columns, width = cpu_weight.shape
    padded = F.pad(cpu_weight, (0, 0, 0, (-columns) % CUBE_ALIGNMENT))
    weight = cpu_weight.npu()
    if columns >= CUBE_ALIGNMENT:
        weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
    cached_bank = torch_npu.npu_format_cast(padded.t().contiguous().unsqueeze(0).npu(), ACL_FORMAT_FRACTAL_NZ)
    padded_weight = None
    if columns % CUBE_ALIGNMENT:
        padded_weight = torch_npu.npu_format_cast(padded.npu(), ACL_FORMAT_FRACTAL_NZ)
    ends = torch.tensor([args.tokens], dtype=torch.int64, device="npu")
    dtype = torch.float16 if args.projection == "input_mix_weight_up" else torch.float32
    inputs = torch.zeros(args.tokens, width, dtype=dtype, device="npu")
    functions = projection_functions(inputs, weight, cached_bank, ends, torch_npu.npu_grouped_matmul, padded_weight)
    if args.arm:
        unknown = set(args.arm) - set(functions)
        if unknown:
            parser.error(f"unknown/ineligible arms: {sorted(unknown)}")
        functions = {
            name: function for name, function in functions.items() if name == "linear_nz_weight" or name in args.arm
        }
    report = {
        "scope": "real_weight_fp16_projection_synthetic_inputs_not_whole_model",
        "projection": args.projection,
        "checkpoint_key": keys[args.projection],
        "tokens": args.tokens,
        "weight_shape": list(cpu_weight.shape),
        "cached_bank_shape": list(cached_bank.shape),
        "cached_bank_storage_bytes": cached_bank.untyped_storage().nbytes(),
        "padded_weight_storage_bytes": padded_weight.untyped_storage().nbytes() if padded_weight is not None else 0,
        "dynamic_casts_included": True,
        "eager_gates": {name: [] for name in functions if name != "linear_nz_weight"},
        "graph_gates": {},
        "skip_batchmatmul_graphs": args.skip_batchmatmul_graphs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.npu.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(1145)
    graphs = {}
    try:
        for scale in (0.1, 0.0, 4.0):
            inputs.copy_((torch.randn(inputs.shape, generator=generator) * scale).to(dtype).npu())
            expected = functions["linear_nz_weight"]()
            for name, function in functions.items():
                if name != "linear_nz_weight":
                    report["eager_gates"][name].append(compare_outputs(expected, function()))
            save()
        qualified = {
            name: function
            for name, function in functions.items()
            if name == "linear_nz_weight" or all(gate_passed(gate) for gate in report["eager_gates"][name])
        }
        report["exact_eager_arms"] = list(qualified)
        if len(qualified) > 1:
            report["eager_timing"] = paired_timing(qualified, 5 if args.tokens >= 2560 else 30, args.repeats)
            if args.tokens <= 128:
                capture_arms = select_graph_arms(qualified, args.skip_batchmatmul_graphs)
                report["graph_skipped_arms"] = sorted(set(qualified) - set(capture_arms))
                save()
                graphs = {name: capture(function) for name, function in capture_arms.items()}
                report["graph_gates"] = {name: [] for name in graphs}
                for scale in (0.1, 0.0, 4.0):
                    inputs.copy_((torch.randn(inputs.shape, generator=generator) * scale).to(dtype).npu())
                    for graph, _ in graphs.values():
                        graph.replay()
                    torch.npu.synchronize()
                    expected = functions["linear_nz_weight"]()
                    for name, (_, output) in graphs.items():
                        report["graph_gates"][name].append(compare_outputs(expected, output))
                    save()
                exact_graphs = {
                    name: graph.replay
                    for name, (graph, _) in graphs.items()
                    if all(gate_passed(gate) for gate in report["graph_gates"][name])
                }
                if len(exact_graphs) > 1:
                    report["graph_timing"] = paired_timing(exact_graphs, 100, args.repeats)
        report["peak_tensor_allocated_bytes"] = torch.npu.max_memory_allocated()
        report["peak_tensor_reserved_bytes"] = torch.npu.max_memory_reserved()
        save()
    except (RuntimeError, ValueError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
