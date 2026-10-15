# SPDX-License-Identifier: Apache-2.0
"""Strict synthetic gates for 310P sorting, reduction schedules and norm/gate fusion."""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.math_probe_cases import PATTERNS


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("sort", "reduction", "gdn_fused", "reduce_sum"), required=True)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sort-groups", type=int, nargs="+", default=[512])
    args = parser.parse_args()
    if any(groups < 512 or groups > 10000 or groups % 8 for groups in args.sort_groups):
        parser.error("sort groups must be multiples of eight in [512, 10000]")
    if args.output.exists():
        parser.error("preserve existing evidence")
    import torch_npu

    from tools.qwen4exp.benchmark_shared_hc_operand_310 import capture, paired_timing

    if not torch_npu.npu.is_available():
        raise RuntimeError("Ascend NPU required")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(args.library))
    name = "qwen_sort_probe_v1" if args.kind == "sort" else "qwen_reduction_probe_v1"
    kernel = torch.classes.qwen_math_next_v1.Kernel(str(args.binary), name)
    launch = torch.ops.qwen_math_next_v1.launch
    report = {"scope": "synthetic_native_math_not_complete_model", "kind": args.kind, "records": []}
    shapes = (
        tuple((rows, groups) for groups in args.sort_groups for rows in (3, 6, 64))
        if args.kind == "sort"
        else ((24, 128), (48, 128), (5120, 128), (3, 2560), (6, 2560), (640, 2560))
    )
    modes = (
        [3]
        if args.kind == "gdn_fused"
        else [0, 1, 2]
        if args.kind == "reduction"
        else [4, 5, 6]
        if args.kind == "reduce_sum"
        else [0]
    )
    for rows, width in shapes:
        if args.kind == "gdn_fused" and width != 128:
            continue
        x = torch.empty(rows, width, dtype=torch.float32, device="npu")
        gate = torch.empty(rows, width, dtype=torch.float16, device="npu")
        gamma = torch.linspace(0.8, 1.2, width, dtype=torch.float32, device="npu")
        for mode in modes:
            geometry = torch.tensor([rows, width, 10 if args.kind == "sort" else mode], dtype=torch.int64, device="npu")

            def candidate(x=x, gate=gate, gamma=gamma, geometry=geometry, mode=mode):
                dtype = torch.int32 if args.kind == "sort" else torch.float16 if mode == 3 else torch.float32
                output = (
                    torch.empty((x.shape[0], 8), dtype=dtype, device=x.device)
                    if args.kind == "reduce_sum"
                    else torch.empty((x.shape[0], 512), dtype=dtype, device=x.device)
                    if args.kind == "sort"
                    else torch.empty_like(x, dtype=dtype)
                )
                arguments = [x, output, geometry] if args.kind == "sort" else [x, gamma, gate, output, geometry]
                launch(kernel, arguments, 8)
                return output[:, :1] if args.kind == "reduce_sum" else output

            def reference(x=x, gate=gate, gamma=gamma, mode=mode):
                if args.kind == "sort":
                    return torch.argsort(x, dim=-1, descending=True, stable=True)[:, :512].int()
                if args.kind == "reduce_sum":
                    return x.square().sum(-1, keepdim=True)
                normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * gamma
                return (normalized * torch.sigmoid(gate)).half() if mode == 3 else normalized

            record = {"rows": rows, "width": width, "mode": mode, "gates": []}
            generator = torch.Generator().manual_seed(7001)
            try:
                for pattern in (*PATTERNS, *(["masked"] if args.kind == "sort" else [])):
                    if pattern == "zeros":
                        host = torch.zeros(rows, width)
                    elif pattern == "ties":
                        host = torch.arange(width).remainder(16).float().expand(rows, width).clone()
                    else:
                        host = torch.randn(rows, width, generator=generator) * (
                            8 if pattern == "large" else 1e-7 if pattern == "near_ties" else 0.1
                        )
                    if pattern == "masked":
                        host[:, width // 2 :] = -float("inf")
                    x.copy_(host.to("npu"))
                    gate.copy_(host.half().to("npu"))
                    checks = compare_outputs(reference(), candidate())
                    record["gates"].append({"pattern": pattern, "checks": checks})
                record["exact"] = all(gate_passed(g["checks"]) for g in record["gates"])
                if record["exact"]:
                    functions = {"baseline": reference, "candidate": candidate}
                    record["eager_timing"] = paired_timing(functions, 5 if rows >= 640 else 20, 3)
                    if rows <= 48:
                        graphs = {name: capture(function) for name, function in functions.items()}
                        x.fill_(0.2)
                        gate.fill_(0.1)
                        for graph, _ in graphs.values():
                            graph.replay()
                        torch.npu.synchronize()
                        record["graph_checks"] = {
                            name: compare_outputs(reference(), output) for name, (_, output) in graphs.items()
                        }
                        record["graph_exact"] = all(gate_passed(g) for g in record["graph_checks"].values())
                        if record["graph_exact"]:
                            record["graph_timing"] = paired_timing(
                                {name: graph.replay for name, (graph, _) in graphs.items()}, 50, 3
                            )
                        del graphs, graph
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                report["records"].append(record)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                raise
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"kind": args.kind, "rows": rows, "width": width, "mode": mode, "exact": record["exact"]}),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
