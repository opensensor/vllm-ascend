# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact parity and timing of joined HC projections; no full server/model load."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing
from tools.qwen4exp.hc_checkpoint import load_hyperconnection
from tools.qwen4exp.resident_candidates import fused_hc_projection as fused
from tools.qwen4exp.resident_candidates import shared_hc_operand as shared
from vllm_ascend.models.qwen4_exp.model import _GatedResidual, _linear
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

DEFAULT_PREFIXES = (
    "model.language_model.layers.0.attn_hyper_connection",
    "model.language_model.layers.0.mlp_hyper_connection",
    "model.language_model.layers.47.mlp_hyper_connection",
    "mtp.layers.0.attn_hyper_connection",
)
HIDDEN_SIZE = 2560
HC_COUNT = 4
LOWRANK = 320


def compare_tensors(expected, actual):
    expected, actual = expected.cpu(), actual.cpu()
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    return {
        "bitwise_equal": torch.equal(expected, actual),
        "finite": finite,
        "different_elements": int(torch.count_nonzero(expected != actual)),
        "max_abs_error": float((expected.float() - actual.float()).abs().max()),
        "actual_sha256": hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
    }


def output_gate(functions):
    outputs = {name: function() for name, function in functions.items()}
    return {
        name: {
            field: compare_tensors(expected, actual)
            for field, expected, actual in zip(("mixed", "combined"), outputs["baseline"], result)
        }
        for name, result in outputs.items()
        if name != "baseline"
    }


def raw_projection_gate(module, values, joined):
    normalized = module._normalize(values)
    actual = F.linear(normalized.to(joined.dtype), joined)
    lowrank = module.input_mix_weight_down.shape[0]
    return {
        "down": compare_tensors(
            _linear(normalized, module.input_mix_weight_down, module.compute_dtype), actual[:, :lowrank]
        ),
        "injection": compare_tensors(
            _linear(normalized, module.block_inject_weight, module.compute_dtype),
            actual[:, lowrank : lowrank + module.hc_count],
        ),
    }


def passed(gate):
    if "bitwise_equal" in gate:
        return gate["bitwise_equal"] and gate["finite"]
    return all(passed(value) for value in gate.values())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="real-weight gate: local checkpoint directory")
    source.add_argument("--synthetic", action="store_true", help="dummy gate only; cannot qualify serving")
    parser.add_argument("--module-prefix", action="append", help="repeat to select checkpoint hyperconnections")
    parser.add_argument("--tokens", type=int, nargs="+", default=[3, 6, 2560])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    if args.repeats < 1 or any(tokens < 1 for tokens in args.tokens):
        parser.error("tokens and repeats must be positive")
    if args.synthetic and args.module_prefix:
        parser.error("module-prefix requires a checkpoint")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(1065)
    report = {
        "scope": "isolated_hyperconnection_mix_combine_synthetic_inputs_not_whole_model",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "real_weights": bool(args.checkpoint),
        "records": [],
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefixes = (args.module_prefix or DEFAULT_PREFIXES) if args.checkpoint else ("synthetic",)
    for prefix in prefixes:
        if args.checkpoint:
            module, keys = load_hyperconnection(args.checkpoint, prefix)
        else:
            keys = {}
            module = _GatedResidual(
                hc_count=HC_COUNT,
                hidden_size=HIDDEN_SIZE,
                lowrank=LOWRANK,
                eps=1e-6,
                params_dtype=torch.float16,
                compute_dtype=torch.float32,
            )
            for parameter in module.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.01)
        module.npu()
        for parameter in module.parameters():
            if parameter.ndim == 2 and min(parameter.shape) >= fused.CUBE_ALIGNMENT:
                parameter.data = torch_npu.npu_format_cast(parameter.data, ACL_FORMAT_FRACTAL_NZ)
        module.prepare_norm_affine()
        pointers = {name: parameter.data_ptr() for name, parameter in module.named_parameters()}
        torch.npu.synchronize()
        before_bytes = torch.npu.memory_allocated()
        joined = fused.prepare_projection(module)
        if joined is None:
            raise RuntimeError(f"{prefix}: expected an eligible FP16 hyperconnection")
        torch.npu.synchronize()
        cache_allocated_bytes = torch.npu.memory_allocated() - before_bytes
        for tokens in args.tokens:
            torch.npu.reset_peak_memory_stats()
            values = torch.randn(tokens, module.hyper_hidden).half().npu()
            block = torch.randn(tokens, module.hidden_size).half().npu()

            def baseline(module=module, values=values, block=block):
                mixed, residual = module.mix(values)
                return mixed, module.combine(block, residual)

            def shared_operand(module=module, values=values, block=block):
                mixed, residual = shared.mix(module, values)
                return mixed, module.combine(block, residual)

            def joint_projection(module=module, values=values, block=block):
                mixed, residual = fused.mix(module, values)
                return mixed, fused.combine(module, block, residual)

            functions = {"baseline": baseline, "shared_operand": shared_operand, "fused_projection": joint_projection}
            record = {
                "prefix": prefix,
                "keys": keys,
                "tokens": tokens,
                "joined_shape": list(joined.shape),
                "cache_storage_bytes": joined.untyped_storage().nbytes(),
                "cache_allocator_bytes": cache_allocated_bytes,
                "saved_auxiliary_bytes": {
                    "baseline": tokens * module.hyper_hidden * 4,
                    "shared_operand": tokens * module.hyper_hidden * 2,
                    "fused_projection": tokens * module.hc_count * 2,
                },
                "eager_gates": [],
                "graph_gates": [],
            }
            for _ in range(3):
                values.copy_(torch.randn_like(values))
                block.copy_(torch.randn_like(block))
                record["eager_gates"].append(
                    {"outputs": output_gate(functions), "projections": raw_projection_gate(module, values, joined)}
                )
            record["passed"] = all(passed(gate) for gate in record["eager_gates"])
            if record["passed"]:
                for function in functions.values():
                    for _ in range(3):
                        function()
                record["eager_timing"] = paired_timing(functions, 2 if tokens >= 2560 else 10, args.repeats)
                if tokens <= 6:
                    captured = {name: capture(function) for name, function in functions.items()}
                    for _ in range(3):
                        values.copy_(torch.randn_like(values))
                        block.copy_(torch.randn_like(block))
                        for graph, _ in captured.values():
                            graph.replay()
                        baseline_outputs = captured["baseline"][1]
                        record["graph_gates"].append(
                            {
                                name: {
                                    field: compare_tensors(expected, actual)
                                    for field, expected, actual in zip(("mixed", "combined"), baseline_outputs, outputs)
                                }
                                for name, (_, outputs) in captured.items()
                                if name != "baseline"
                            }
                        )
                    record["passed"] = all(passed(gate) for gate in record["graph_gates"])
                    if record["passed"]:
                        record["graph_timing"] = paired_timing(
                            {name: graph.replay for name, (graph, _) in captured.items()}, 100, args.repeats
                        )
                    del captured, graph, baseline_outputs
            if pointers != {name: parameter.data_ptr() for name, parameter in module.named_parameters()}:
                raise RuntimeError("checkpoint parameter storage changed")
            report["passed"] &= record["passed"]
            record["peak_tensor_allocated_bytes"] = torch.npu.max_memory_allocated()
            record["peak_tensor_reserved_bytes"] = torch.npu.max_memory_reserved()
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"prefix": prefix, "tokens": tokens, "passed": record["passed"]}), flush=True)
            del functions, baseline, shared_operand, joint_projection, values, block
        fused.release_projection(module)
        del joined, module
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
