# SPDX-License-Identifier: Apache-2.0
"""Measure isolated real TP2 middle-layer components and projection layouts.

This does not measure complete-model speed: collectives and routed experts are
excluded. Run only on the assigned sidecard and retain the JSON receipt.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from .layer_math import TextLayer, mhc_post, mhc_pre, prepare_projection_layout
from .layer_weights import load_dense

PROFILE_LAYERS = (18, 19)
WARMUP_STEPS = 5
MEASURED_STEPS = 20
FRACTAL_NZ = 29
ND = 2


def measure(operation, synchronize):
    for _ in range(WARMUP_STEPS):
        operation()
    synchronize()
    samples = []
    for _ in range(MEASURED_STEPS):
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def run(checkpoint, device, output, ops_library):
    import torch
    import torch_npu

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("profile requires the assigned sidecard visibility")
    torch.npu.set_device(int(device.split(":")[1]))
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.ops.load_library(ops_library)
    sinkhorn = torch.ops._C_ascend.mhc_sinkhorn_310
    config = json.loads((checkpoint / "config.json").read_bytes())["text_config"]
    weights, _ = load_dense(checkpoint, PROFILE_LAYERS, device, torch, tp_rank=0)
    cases = []
    torch.manual_seed(20261014)
    with torch.inference_mode():
        for index in PROFILE_LAYERS:
            layer = TextLayer(
                index, weights[index], config, device, torch, None, 2048, native_sinkhorn=sinkhorn, tp_rank=0
            )
            prepare_projection_layout(layer.weights, lambda value, _: torch_npu.npu_format_cast(value, ND))
            for rows in (1, 32):
                x = torch.randn(rows, 4096).half().to(device)
                residual = torch.randn(rows, 4, 4096).to(device)
                post, comb, _ = mhc_pre(
                    residual, layer.weights, "attn", "input_layernorm.weight", config, torch, sinkhorn
                )
                components = {
                    "mhc_pre": lambda residual=residual, layer=layer: mhc_pre(
                        residual, layer.weights, "attn", "input_layernorm.weight", config, torch, sinkhorn
                    ),
                    "mhc_post": lambda x=x, residual=residual, post=post, comb=comb: mhc_post(
                        x, residual, post, comb, torch
                    ),
                    "attention": lambda layer=layer, x=x: (
                        layer.kda(x) if layer.linear else (layer.reset(), layer.dsa(x))[1]
                    ),
                    "shared_mlp": lambda layer=layer, x=x: layer.dense_mlp(x, "mlp.shared_experts"),
                    "router": lambda x=x, layer=layer: (
                        x.float()[:, None, :] * layer.weights["mlp.gate.weight"][None]
                    ).sum(-1),
                }
                for kind, operation in components.items():
                    cases.append(dict(layer=index, rows=rows, kind=kind, **measure(operation, torch.npu.synchronize)))
                for name, weight in list(layer.weights.items()):
                    if weight.ndim != 2 or weight.dtype != torch.float16 or "indexer" in name or "kv_b" in name:
                        continue
                    if not (
                        name.endswith("proj.weight")
                        or name == "self_attn.in_proj"
                        or name == "self_attn.kv_a_proj_with_mqa.weight"
                    ):
                        continue
                    projected_input = torch.randn(rows, weight.shape[1]).half().to(device)
                    plain = lambda projected_input=projected_input, weight=weight: torch.nn.functional.linear(
                        projected_input, weight
                    )
                    formatted = torch_npu.npu_format_cast(weight, FRACTAL_NZ)
                    nz = lambda projected_input=projected_input, formatted=formatted: torch.nn.functional.linear(
                        projected_input, formatted
                    )
                    expected, actual = plain(), nz()
                    error = (expected.float() - actual.float()).abs().cpu()
                    cases.append(
                        dict(
                            layer=index,
                            rows=rows,
                            kind="projection",
                            name=name,
                            shape=list(weight.shape),
                            max_abs_error=float(error.max()),
                            bitwise_equal=bool(torch.equal(expected.cpu(), actual.cpu())),
                            plain_layout=measure(plain, torch.npu.synchronize),
                            blocked_layout=measure(nz, torch.npu.synchronize),
                        )
                    )
    report = {
        "device": device,
        "scope": "Isolated actual-weight TP2 rank0 components; no EP or collectives",
        "complete_model_speed_validated": False,
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("npu:0", "npu:1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ops-library", required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.output, args.ops_library)


if __name__ == "__main__":
    main()
