# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate 16-row HC gate chains with real weights and changed-input graphs.

Includes padding, casting, both GEMMs and activation operations in timings.
Complete mix/combine gates retain the original four-row injection projection.
No model server, collectives, persistent scratch state or new OPP is needed.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing
from tools.qwen4exp.resident_candidates import hc_projection_layout as candidate

DEFAULT_PREFIXES = (
    "model.language_model.layers.0.attn_hyper_connection",
    "model.language_model.layers.0.mlp_hyper_connection",
    "model.language_model.layers.47.mlp_hyper_connection",
    "mtp.layers.0.attn_hyper_connection",
)
PATTERN_SCALES = (0.0, 0.1, 1.0, 4.0)


def probe_functions(module, values, block):
    """Each closure reads current inputs; no computed input is cached."""

    def baseline_gate():
        normalized = module._normalize(values)
        gate = F.silu(F.linear(normalized.half(), module.input_mix_weight_down) / module.hc_count)
        return torch.sigmoid(F.linear(gate, module.input_mix_weight_up))

    def padded_gate():
        return candidate.padded_gate(
            module._normalize(values), module.input_mix_weight_down, module.input_mix_weight_up, module.hc_count
        )

    def baseline_complete():
        mixed, residual = module.mix(values)
        return mixed, module.combine(block, residual)

    def padded_complete():
        mixed, residual = candidate.mix(module, values)
        return mixed, module.combine(block, residual)

    return {
        "gate_chain": {"baseline": baseline_gate, "padded": padded_gate},
        "mix_combine": {"baseline": baseline_complete, "padded": padded_complete},
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--module-prefix", action="append")
    parser.add_argument("--tokens", type=int, nargs="+", default=list(candidate.DECODE_ROWS))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or any(rows not in candidate.DECODE_ROWS for rows in args.tokens):
        parser.error("use a new output path, positive repeats, and 3/6-token shapes")
    # Isolated hardware process only: importing this module for host tests does
    # not initialize torch_npu or checkpoint readers.
    import torch_npu

    from tools.qwen4exp.hc_checkpoint import load_hyperconnection

    torch.npu.set_device(args.device)
    if "310" not in torch.npu.get_device_name(args.device):
        raise RuntimeError("this probe targets Ascend 310P")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(1410)
    report = {
        "scope": "real_checkpoint_hc_gate_chain_and_mix_combine_not_whole_model",
        "checkpoint": str(args.checkpoint),
        "candidate_hardware_qualified": False,
        "row_padding_included_in_timing": True,
        "changed_weight_layout": False,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for prefix in args.module_prefix or DEFAULT_PREFIXES:
            module, keys = load_hyperconnection(args.checkpoint, prefix)
            module = module.to(device=f"npu:{args.device}")
            # Match serving: learned matrix weights with >=16 rows are NZ;
            # tiny four-row injection retains its original ND representation.
            for weight in module.parameters():
                if weight.ndim == 2 and min(weight.shape) >= candidate.CUBE_ROWS:
                    weight.data = torch_npu.npu_format_cast(weight.data, 29)
            module.prepare_norm_affine()
            for rows in args.tokens:
                values = torch.zeros(rows, candidate.HC_WIDTH, device=module.input_mix_weight_down.device).half()
                block = torch.zeros(rows, module.hidden_size, device=values.device).half()
                if not candidate.eligible(module, module._normalize(values)):
                    raise ValueError(f"candidate would fall back for {prefix}, tokens={rows}")
                generator = torch.Generator().manual_seed(1411)

                def refresh(scale, values=values, block=block, generator=generator):
                    values.copy_((torch.randn(values.shape, generator=generator) * scale).half().to(values.device))
                    block.copy_((torch.randn(block.shape, generator=generator) * scale).half().to(block.device))

                for scope, functions in probe_functions(module, values, block).items():
                    record = {"prefix": prefix, "checkpoint_keys": keys, "tokens": rows, "scope": scope, "eager": []}
                    report["records"].append(record)
                    for scale in PATTERN_SCALES:
                        refresh(scale)
                        record["eager"].append(compare_outputs(functions["baseline"](), functions["padded"]()))
                        save()
                    if not all(gate_passed(gate) for gate in record["eager"]):
                        record["decision"] = "rejected_eager_parity"
                        save()
                        continue
                    captured = {name: capture(function) for name, function in functions.items()}
                    record["graph"] = []
                    for scale in reversed(PATTERN_SCALES):
                        refresh(scale)
                        for graph, _ in captured.values():
                            graph.replay()
                        torch.npu.synchronize()
                        expected = functions["baseline"]()
                        record["graph"].append(
                            {name: compare_outputs(expected, output) for name, (_, output) in captured.items()}
                        )
                        save()
                    exact = all(gate_passed(gate) for trial in record["graph"] for gate in trial.values())
                    if exact:
                        record["eager_timing"] = paired_timing(functions, 20, args.repeats)
                        record["graph_timing"] = paired_timing(
                            {name: graph.replay for name, (graph, _) in captured.items()}, 100, args.repeats
                        )
                        record["decision"] = "component_exact_timing_recorded"
                    else:
                        record["decision"] = "rejected_graph_parity"
                    torch.npu.synchronize()
                    del captured
                    save()
        report["candidate_hardware_qualified"] = all(
            record.get("decision") == "component_exact_timing_recorded" for record in report["records"]
        )
        save()
    except (RuntimeError, ValueError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
