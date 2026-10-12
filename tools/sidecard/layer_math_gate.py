# SPDX-License-Identifier: Apache-2.0
"""Real dense-weight component parity before whole-model serving promotion."""

import argparse
import json
import os
import time
from pathlib import Path

from .layer_math import TextLayer, recurrent_kda
from .layer_weights import load_dense


def check(actual, expected, *, rtol, atol, torch):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    error = (actual - expected).abs()
    okay = bool(torch.isfinite(actual).all()) and bool((error <= atol + rtol * expected.abs()).all())
    return {"passed": okay, "max_abs_error": float(error.max()), "rtol": rtol, "atol": atol, "elements": actual.numel()}


def run(root, device, output, ops_library=None):
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if device.startswith("npu"):
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
            raise ValueError("component gate must preserve the other Ascend card")
        import torch_npu

        torch.npu.set_device(int(device.split(":")[1]))
        torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(2)
    config = json.loads((root / "config.json").read_bytes())["text_config"]
    sinkhorn = None
    if ops_library and device.startswith("npu"):
        torch.ops.load_library(ops_library)
        sinkhorn = torch.ops._C_ascend.mhc_sinkhorn_310
    cpu_weights, _ = load_dense(root, (0, 3), "cpu", torch)
    weights = {
        index: {name: value.to(device).clone() for name, value in row.items()} for index, row in cpu_weights.items()
    }
    layers = {
        index: TextLayer(index, weights[index], config, device, torch, None, 2048, native_sinkhorn=sinkhorn)
        for index in (0, 3)
    }
    reference = {index: TextLayer(index, cpu_weights[index], config, "cpu", torch, None, 2048) for index in (0, 3)}
    torch.manual_seed(20261012)
    cases = []
    with torch.inference_mode():
        for phase in range(2):
            for rows in (1, 4, 32):
                hidden = torch.randn(rows, 4096).half() * (0.5 + phase)
                for layer in (*layers.values(), *reference.values()):
                    layer.reset()
                begin = time.perf_counter()
                actual = layers[0].forward(hidden.to(device))
                actual = [value.cpu() for value in actual]
                elapsed = time.perf_counter() - begin
                expected = reference[0].forward(hidden)
                checks = [
                    check(value, golden, rtol=0.025, atol=0.02, torch=torch)
                    for value, golden in zip(actual, expected, strict=True)
                ]
                cases.append(
                    dict(
                        kind="real_dense_layer0_complete_mhc_kda_mlp",
                        phase=phase,
                        rows=rows,
                        checks=checks,
                        passed=all(value["passed"] for value in checks),
                        wall_seconds=elapsed,
                    )
                )
                actual = layers[3].dsa(hidden.to(device)).cpu()
                expected = reference[3].dsa(hidden)
                result = check(actual, expected, rtol=0.025, atol=0.02, torch=torch)
                cases.append(dict(kind="real_dsa_layer3_attention", phase=phase, rows=rows, **result))
        if device == "cuda":
            from .cuda_layer_ops import recurrent

            for phase in range(2):
                q, k, v = [torch.randn(32, 64, 128).half() * (phase + 0.5) for _ in range(3)]
                gate, beta = -torch.rand_like(q.float()), torch.rand(32, 64)
                state = torch.randn(64, 128, 128)
                golden_state = state.clone()
                expected = recurrent_kda(q, k, v, gate, beta, golden_state, torch)
                actual_state = state.to(device)
                actual = recurrent(
                    q.to(device), k.to(device), v.to(device), gate.to(device), beta.to(device), actual_state
                )
                values = [
                    check(actual, expected, rtol=0.001, atol=0.0005, torch=torch),
                    check(actual_state, golden_state, rtol=1e-4, atol=1e-5, torch=torch),
                ]
                cases.append(
                    dict(
                        kind="cuda_fused_kda_full_output_and_state",
                        phase=phase,
                        checks=values,
                        passed=all(value["passed"] for value in values),
                    )
                )
    report = dict(
        device=device,
        cases=cases,
        passed=all(case["passed"] for case in cases),
        complete_model_speed_validated=False,
        scope=(
            "Real layer0 weights and DSA layer3; changed synthetic inputs; "
            "CPU arithmetic comparison; no complete MoE layer"
        ),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"device": device, "passed": report["passed"], "cases": len(cases)}), flush=True)
    if not report["passed"]:
        raise RuntimeError("dense layer component parity failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "npu:0", "npu:1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ops-library")
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.output, args.ops_library)


if __name__ == "__main__":
    main()
