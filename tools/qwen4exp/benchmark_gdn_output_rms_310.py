# SPDX-License-Identifier: Apache-2.0
"""Real-weight TP-rank-zero GDN output comparison without a server or HCCL."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.resident_candidates.gdn_output_rms import make_project_output, project_output
from vllm_ascend.models.qwen4_exp.model import _GDNAttention


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 46])
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--gate-probabilities", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence")
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing

    torch.npu.set_device(0)
    replacement = project_output
    if bool(args.native_library) != bool(args.native_binary):
        parser.error("native library and binary must be supplied together")
    if args.native_library:
        from tools.qwen4exp.direct_gdn_output import DirectGDNNormGate

        torch.ops.load_library(str(args.native_library))
        replacement = make_project_output(
            DirectGDNNormGate(str(args.native_binary)), gate_probabilities=args.gate_probabilities
        )
    torch_npu.npu.set_compile_mode(jit_compile=False)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = sorted({k.removesuffix(".norm.weight") for k in index if ".linear_attn.norm.weight" in k})
    selected = [prefix for prefix in prefixes if int(prefix.split(".layers.")[1].split(".")[0]) in args.layers]
    if len(selected) != len(set(args.layers)):
        parser.error("requested GDN layers missing from checkpoint")
    report = {"scope": "real_GDN_output_projection_TP_rank0_no_collective_not_model_quality", "records": []}
    for prefix in selected:
        weights = {}
        for attr, suffix in (
            ("norm_weight", "norm.weight"),
            ("in_proj_z", "in_proj_z.weight"),
            ("out_proj", "out_proj.weight"),
        ):
            key = f"{prefix}.{suffix}"
            with safe_open(args.model / index[key], framework="pt", device="cpu") as reader:
                weights[attr] = reader.get_tensor(key).half()
        width = weights["norm_weight"].numel()
        value_dim = weights["in_proj_z"].shape[0] // 4
        weights["in_proj_z"] = weights["in_proj_z"][:value_dim].contiguous()
        weights["out_proj"] = weights["out_proj"][:, :value_dim].contiguous()
        module = SimpleNamespace(
            params=SimpleNamespace(head_v_dim=width),
            num_v_heads=value_dim // width,
            value_dim=value_dim,
            compute_dtype=torch.float32,
            params_dtype=torch.float16,
            rms_norm_eps=1e-6,
            tp_size=1,
            _tp_reduce=None,
            **{k: v.npu() for k, v in weights.items()},
        )
        for tokens in (3, 6, 640):
            hidden = module.in_proj_z.shape[1]
            inputs = torch.zeros(tokens, hidden, dtype=torch.float16, device="npu")
            out = torch.zeros(tokens, module.num_v_heads, width, dtype=torch.float32, device="npu")
            functions = {
                "baseline": lambda inputs=inputs, out=out, module=module: _GDNAttention._project_output(
                    module, inputs, out
                ),
                "candidate": lambda inputs=inputs, out=out, module=module: replacement(module, inputs, out),
            }
            record = {"prefix": prefix, "tokens": tokens, "heads": module.num_v_heads, "width": width, "gates": []}
            generator = torch.Generator().manual_seed(2105)
            for pattern in ("random", "zero", "large", "near_zero"):
                scale = 8 if pattern == "large" else 1e-7 if pattern == "near_zero" else 0 if pattern == "zero" else 0.1
                inputs.copy_((torch.randn(inputs.shape, generator=generator) * scale).half().npu())
                out.copy_((torch.randn(out.shape, generator=generator) * scale).npu())
                record["gates"].append(
                    {"pattern": pattern, "checks": compare_outputs(functions["baseline"](), functions["candidate"]())}
                )
            record["exact"] = all(gate_passed(g["checks"]) for g in record["gates"])
            if record["exact"]:
                record["eager_timing"] = paired_timing(functions, 5 if tokens >= 640 else 20, 3)
                if tokens <= 6:
                    graphs = {k: capture(fn) for k, fn in functions.items()}
                    inputs.fill_(0.2)
                    out.fill_(0.3)
                    for graph, _ in graphs.values():
                        graph.replay()
                    torch.npu.synchronize()
                    checks = {k: compare_outputs(functions["baseline"](), output) for k, (_, output) in graphs.items()}
                    record["graph_checks"] = checks
                    record["graph_exact"] = all(gate_passed(v) for v in checks.values())
                    if record["graph_exact"]:
                        record["graph_timing"] = paired_timing({k: g.replay for k, (g, _) in graphs.items()}, 50, 3)
                    del graphs, graph
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"prefix": prefix, "tokens": tokens, "exact": record["exact"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
