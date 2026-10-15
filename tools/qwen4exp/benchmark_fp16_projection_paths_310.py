# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare exact FP16 projection/layout paths; include dynamic operand casts."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECTIONS = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
DEFAULT_PREFIX = "model.language_model.layers.0.attn_hyper_connection"
DYNAMIC_FORMAT_ARMS = ("linear_nz_input", "mm_nz_input_cached_transpose")


def operator_functions(inputs, weight, cached_transpose, format_input):
    """Pack dynamic inputs inside each call; only learned weights are cached."""

    def operand():
        return inputs.to(weight.dtype)

    return {
        "linear_nz_weight": lambda: F.linear(operand(), weight),
        "linear_nz_input": lambda: F.linear(format_input(operand()), weight),
        "mm_transpose_view": lambda: torch.mm(operand(), weight.t()),
        "mm_cached_transpose": lambda: torch.mm(operand(), cached_transpose),
        "mm_nz_input_cached_transpose": lambda: torch.mm(format_input(operand()), cached_transpose),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--module-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--tokens", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--projections", choices=PROJECTIONS, nargs="+", default=list(PROJECTIONS))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--skip-dynamic-format-graphs",
        action="store_true",
        help="qualified CANN rejects dynamic format casts as aclop Identity during capture",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or any(tokens < 1 for tokens in args.tokens):
        parser.error("use a new output path and positive tokens/repeats")
    # Isolated worker-only imports; host tests exercise function construction
    # without initializing an NPU runtime or importing the adapter utilities.
    import torch_npu

    from tools.qwen4exp.benchmark_fused_hc_projection_310 import compare_tensors
    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing
    from tools.qwen4exp.hc_checkpoint import load_hyperconnection
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(1075)
    module, keys = load_hyperconnection(args.checkpoint, args.module_prefix)
    report = {
        "scope": "real_checkpoint_fp16_projections_synthetic_inputs_not_whole_model",
        "checkpoint": str(args.checkpoint),
        "prefix": args.module_prefix,
        "dynamic_casts_included": True,
        "cached_transpose_preparation_excluded": True,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def format_input(tensor):
        return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)

    for name in args.projections:
        checkpoint_weight = getattr(module, name).detach()
        weight = format_input(checkpoint_weight.npu())
        # Cache the actual physical transpose; passing an NZ tensor's .t()
        # view is a separate arm, since CANN may select a different kernel.
        cached_transpose = format_input(checkpoint_weight.t().contiguous().npu())
        for tokens in args.tokens:
            torch.npu.reset_peak_memory_stats()
            input_dtype = torch.float16 if name == "input_mix_weight_up" else torch.float32
            inputs = torch.randn(tokens, weight.shape[1], dtype=input_dtype, device="npu")
            functions = operator_functions(inputs, weight, cached_transpose, format_input)
            record = {
                "projection": name,
                "key": keys[name],
                "tokens": tokens,
                "weight_shape": list(weight.shape),
                "input_dtype": str(input_dtype),
                "cached_transpose_storage_bytes": cached_transpose.untyped_storage().nbytes(),
                "eager_gates": {arm: [] for arm in functions if arm != "linear_nz_weight"},
                "graph_gates": {},
            }
            for _ in range(3):
                inputs.copy_(torch.randn_like(inputs))
                expected = functions["linear_nz_weight"]()
                for arm, function in functions.items():
                    if arm != "linear_nz_weight":
                        record["eager_gates"][arm].append(compare_tensors(expected, function()))
            eligible = {
                arm: function
                for arm, function in functions.items()
                if arm == "linear_nz_weight"
                or all(gate["bitwise_equal"] and gate["finite"] for gate in record["eager_gates"][arm])
            }
            for function in eligible.values():
                for _ in range(3):
                    function()
            record["eager_timing"] = paired_timing(eligible, 2 if tokens >= 2560 else 10, args.repeats)
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if tokens <= 6:
                record["skipped_graph_arms"] = (
                    {
                        arm: "dynamic npu_format_cast invokes uncapturable aclop Identity on the qualified stack"
                        for arm in DYNAMIC_FORMAT_ARMS
                        if arm in eligible
                    }
                    if args.skip_dynamic_format_graphs
                    else {}
                )
                captured = {
                    arm: capture(function)
                    for arm, function in eligible.items()
                    if arm not in record["skipped_graph_arms"]
                }
                record["graph_gates"] = {arm: [] for arm in captured if arm != "linear_nz_weight"}
                for _ in range(3):
                    inputs.copy_(torch.randn_like(inputs))
                    for graph, _ in captured.values():
                        graph.replay()
                    expected = captured["linear_nz_weight"][1]
                    for arm, (_, output) in captured.items():
                        if arm != "linear_nz_weight":
                            record["graph_gates"][arm].append(compare_tensors(expected, output))
                graph_functions = {
                    arm: graph.replay
                    for arm, (graph, _) in captured.items()
                    if arm == "linear_nz_weight"
                    or all(gate["bitwise_equal"] and gate["finite"] for gate in record["graph_gates"][arm])
                }
                record["graph_timing"] = paired_timing(graph_functions, 100, args.repeats)
                del captured, graph_functions, graph, output
            record["peak_tensor_allocated_bytes"] = torch.npu.max_memory_allocated()
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"projection": name, "tokens": tokens, "exact_eager_arms": list(eligible)}), flush=True)
            del functions, eligible, function, expected, inputs
        del weight, cached_transpose


if __name__ == "__main__":
    main()
