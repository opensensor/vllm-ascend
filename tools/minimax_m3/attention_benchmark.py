# SPDX-License-Identifier: Apache-2.0
"""Compare attention implementations with real-layer projections and CPU parity."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from tools.minimax_m3.layer_probe import attention, digest, metrics, project


def load_ops(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:4")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    baseline = load_ops(args.baseline, "m3_attention_baseline")
    candidate = load_ops(args.candidate, "m3_attention_candidate")
    weights = load_file(args.bundle / "weights.safetensors")
    config = json.loads((args.bundle / "config.json").read_text())
    reports = []
    with torch.inference_mode():
        for length, queries in ((129, 128), (512, 128), (2048, 128), (2048, 1), (2049, 1)):
            hidden = torch.randn(length, config["hidden_size"], generator=torch.Generator().manual_seed(93)).half()
            projections = project(weights, hidden, config, queries)
            # One TP2 rank owns half the Q/KV heads; index K stays replicated.
            rank_config = {
                **config,
                "num_attention_heads": config["num_attention_heads"] // 2,
                "num_key_value_heads": config["num_key_value_heads"] // 2,
            }
            for name in ("q", "k", "v", "index_q"):
                projections[name] = projections[name][:, : projections[name].shape[1] // 2].contiguous()
            reference = attention(baseline, projections, rank_config, length, queries)
            device_projections = {k: v.to(args.device) for k, v in projections.items()}
            variants = {}
            previous = None
            for name, module in (("baseline", baseline), ("candidate", candidate)):
                for _ in range(2):
                    attention(module, device_projections, rank_config, length, queries)
                torch.npu.synchronize()
                durations = []
                for _ in range(args.iterations):
                    started = time.perf_counter()
                    result = attention(module, device_projections, rank_config, length, queries)
                    torch.npu.synchronize()
                    durations.append(time.perf_counter() - started)
                variants[name] = {
                    "median_seconds": statistics.median(durations),
                    "durations_seconds": durations,
                    "cpu_comparison": metrics(result, reference),
                }
                if previous is not None:
                    variants[name]["baseline_device_comparison"] = metrics(result, previous)
                previous = result.cpu()
            case = {"length": length, "queries": queries, "variants": variants}
            reports.append(case)
            print(json.dumps(case), flush=True)
    report = {
        "scope": "TP2 head geometry, real layer weights with seeded activations; includes cache setup, no collectives",
        "baseline_sha256": digest(args.baseline),
        "candidate_sha256": digest(args.candidate),
        "weights_sha256": digest(args.bundle / "weights.safetensors"),
        "cases": reports,
        "passed": all(
            v["cpu_comparison"]["finite"] and v["cpu_comparison"]["nrmse"] < 0.001
            for case in reports
            for v in case["variants"].values()
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
