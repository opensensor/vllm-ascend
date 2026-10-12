# SPDX-License-Identifier: Apache-2.0
"""Real full-head CUDA middle attention/shared/routing against CPU references."""

import argparse
import json
from pathlib import Path

from .layer_math import TextLayer, mhc_post, mhc_pre
from .layer_math_gate import check
from .layer_weights import load_dense


def run(checkpoint, output):
    """Gate complete head dimensions and BF16-range mHC on changed inputs."""
    import torch

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    config = json.loads((checkpoint / "config.json").read_bytes())["text_config"]
    host, _ = load_dense(checkpoint, (18, 19), "cpu", torch)
    device = {index: {name: value.cuda().clone() for name, value in weights.items()} for index, weights in host.items()}
    cpu = {index: TextLayer(index, host[index], config, "cpu", torch, None, 2048) for index in host}
    gpu = {index: TextLayer(index, device[index], config, "cuda", torch, None, 2048) for index in host}
    cases = []
    with torch.inference_mode():
        for rows in (1, 4, 32):
            for phase in (0, 1):
                torch.manual_seed(72611 + rows + phase)
                x = (torch.randn(rows, 4096) * (phase + 0.5)).half()
                residual = (torch.randn(rows, 4, 4096) * (70000 if phase else 1)).bfloat16().float()
                for index in host:
                    actual, golden = gpu[index], cpu[index]
                    actual.reset()
                    golden.reset()
                    for kind in ("attention", "shared", "mhc", "router"):
                        if kind == "attention":
                            a = actual.kda(x.cuda()) if actual.linear else actual.dsa(x.cuda())
                            e = golden.kda(x) if golden.linear else golden.dsa(x)
                            checks = [check(a, e, rtol=0.025, atol=0.02, torch=torch)]
                            if actual.linear:
                                checks.append(check(actual.state, golden.state, rtol=0.025, atol=0.02, torch=torch))
                        elif kind == "shared":
                            checks = [
                                check(
                                    actual.dense_mlp(x.cuda(), "mlp.shared_experts"),
                                    golden.dense_mlp(x, "mlp.shared_experts"),
                                    rtol=0.025,
                                    atol=0.02,
                                    torch=torch,
                                )
                            ]
                        elif kind == "mhc":
                            a = mhc_pre(
                                residual.cuda(), actual.weights, "attn", "input_layernorm.weight", config, torch
                            )
                            e = mhc_pre(residual, golden.weights, "attn", "input_layernorm.weight", config, torch)
                            checks = [
                                check(v, r, rtol=0.025, atol=0.02, torch=torch) for v, r in zip(a, e, strict=True)
                            ]
                            checks.append(
                                check(
                                    mhc_post(x.cuda(), residual.cuda(), a[0], a[1], torch),
                                    mhc_post(x, residual, e[0], e[1], torch),
                                    rtol=0.025,
                                    atol=0.02,
                                    torch=torch,
                                )
                            )
                        else:
                            a = x.cuda().float() @ actual.weights["mlp.gate.weight"].t()
                            e = x.float() @ golden.weights["mlp.gate.weight"].t()
                            checks = [check(a, e, rtol=1e-4, atol=1e-4, torch=torch)]
                            a_ids = torch.topk(
                                torch.sigmoid(a) + actual.weights["mlp.gate.e_score_correction_bias"], 8, -1
                            ).indices.cpu()
                            e_ids = torch.topk(
                                torch.sigmoid(e) + golden.weights["mlp.gate.e_score_correction_bias"], 8, -1
                            ).indices
                            checks.append({"passed": torch.equal(a_ids, e_ids), "exact_router_ids": True})
                        cases.append(
                            {
                                "layer": index,
                                "rows": rows,
                                "phase": phase,
                                "kind": kind,
                                "checks": checks,
                                "passed": all(c["passed"] for c in checks),
                            }
                        )
    report = {
        "passed": all(c["passed"] for c in cases),
        "real_checkpoint_weights": True,
        "tensor_parallel": False,
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError("CUDA middle backbone CPU-reference gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
