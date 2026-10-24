# SPDX-License-Identifier: Apache-2.0
"""Bounded real M3.1 expert conversion and installed grouped-W4 device gate.

One real expert is repeated across 64 slots to exercise production bank shapes.
This does not qualify an entire MoE bank or the model's language quality.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tools.glm_perf.glm_int4 import pack_nz_codes, unpack_canonical_codes
from tools.glm_perf.operator_bench import nzpacked_codes
from tools.minimax_m3.layer_probe import digest, metrics

LOCAL_EXPERTS = 64
MAX_ROUTES = 128
CHUNK_ROWS = 256
MAX_PROJECTION_NRMSE = 0.003
ACTIVATION_TOKENS = 4


def prepare(model, cache, output, layer, expert):
    # Conversion dependencies are unnecessary in an isolated device process.
    from tools.minimax_m3.mixed_checkpoint import REVISION, MixedCheckpoint
    from tools.minimax_m3.probe_w4 import quantize_chunk

    source = MixedCheckpoint(model, cache)
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}."
    tensors, quality = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for projection in ("w1", "w3", "w2"):
        weight = source.weight(base + projection + ".weight")
        codes, scales = [], []
        error, norm = 0.0, 0.0
        for start in range(0, len(weight), CHUNK_ROWS):
            chunk = weight[start : start + CHUNK_ROWS]
            packed, scale, reconstructed = quantize_chunk(chunk, "mse", mse_backend="numba")
            codes.append(packed)
            scales.append(scale)
            error += (reconstructed - chunk).double().square().sum().item()
            norm += chunk.double().square().sum().item()
        tensors[projection + "_codes"] = torch.cat(codes)
        tensors[projection + "_scales"] = torch.cat(scales)
        tensors[projection + "_reference"] = weight.half()
        quality[projection] = {"shape": list(weight.shape), "nrmse": (error / norm) ** 0.5}
    save_file(tensors, output / "weights.safetensors")
    (output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps({"revision": REVISION, "layer": layer, "expert": expert}, indent=2) + "\n"
    )


def run(bundle, ops, binding, device, routes=MAX_ROUTES, layout="canonical"):
    if routes < LOCAL_EXPERTS or routes % LOCAL_EXPERTS:
        raise ValueError("Probe route count must be a positive multiple of the local expert count")
    torch.npu.set_device(device)
    spec = importlib.util.spec_from_file_location("m3_probe_ops", ops)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = load_file(bundle / "weights.safetensors")
    torch.ops.load_library(str(binding))
    operator = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    report = {
        "device": device,
        "binding_sha256": digest(binding),
        "ops_sha256": digest(ops),
        "weights_sha256": digest(bundle / "weights.safetensors"),
        "probe_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "local_experts": LOCAL_EXPERTS,
        "routes": routes,
        "layout": layout,
        "bank_contents": "One real converted expert repeated in every slot",
        "projections": {},
    }
    for projection in ("w1", "w3", "w2"):
        codes = weights[projection + "_codes"]
        scales = weights[projection + "_scales"]
        reference = module.unpack_w4(codes, scales).half().float()
        if layout == "nz128":
            codes = nzpacked_codes(codes[None], reference.shape[1], 4)[0].view(torch.int8)
        elif layout == "nz256":
            codes = pack_nz_codes(unpack_canonical_codes(codes, reference.shape[1])[None])[0]
        bank_codes = codes.to(device)[None].repeat(LOCAL_EXPERTS, 1, 1).contiguous()
        bank_scales = scales.float().to(device)[None].repeat(LOCAL_EXPERTS, 1, 1).contiguous()
        x = torch.randn(routes, reference.shape[1], generator=torch.Generator().manual_seed(31)).half()
        cpu = (x.float() @ reference.T).half()
        device_x = x.to(device)
        results = {}
        for case, ends in (
            ("equal_routes_each", torch.arange(1, LOCAL_EXPERTS + 1) * (routes // LOCAL_EXPERTS)),
            ("last_expert_only", torch.tensor([0] * (LOCAL_EXPERTS - 1) + [routes])),
            ("zero_local_routes", torch.zeros(LOCAL_EXPERTS, dtype=torch.int64)),
        ):
            result = operator(device_x, bank_codes, bank_scales, ends.to(device=device, dtype=torch.int64))
            results[case] = metrics(result, torch.zeros_like(cpu) if case == "zero_local_routes" else cpu)
        report["projections"][projection] = results
        del bank_codes, bank_scales
    x = torch.randn(
        ACTIVATION_TOKENS, weights["w1_reference"].shape[1], generator=torch.Generator().manual_seed(37)
    ).half()
    reference_weights = {
        p: module.unpack_w4(weights[p + "_codes"], weights[p + "_scales"]).half().float() for p in ("w1", "w3", "w2")
    }
    projected = {p: (x.float() @ reference_weights[p].T).half() for p in ("w1", "w3")}
    cpu_activation = module.clipped_swiglu(projected["w1"], projected["w3"]).half()
    npu_activation = module.clipped_swiglu(projected["w1"].to(device), projected["w3"].to(device)).half()
    report["clipped_swiglu_identical_inputs"] = metrics(npu_activation, cpu_activation)
    cpu_down = (cpu_activation.float() @ reference_weights["w2"].T).half()
    source = {p: weights[p + "_reference"].float() for p in ("w1", "w3", "w2")}
    source_gate = (x.float() @ source["w1"].T).half()
    source_up = (x.float() @ source["w3"].T).half()
    source_down = (module.clipped_swiglu(source_gate, source_up).half().float() @ source["w2"].T).half()
    report["requantized_expert_vs_source_synthetic_inputs"] = metrics(cpu_down, source_down)
    report["numerical_gate_passed"] = all(
        result["finite"] and result["nrmse"] <= MAX_PROJECTION_NRMSE
        for cases in report["projections"].values()
        for result in cases.values()
    )
    report["numerical_gate_passed"] &= (
        report["clipped_swiglu_identical_inputs"]["finite"]
        and report["clipped_swiglu_identical_inputs"]["nrmse"] <= MAX_PROJECTION_NRMSE
    )
    report["full_bank_or_service_validated"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ops", type=Path, default=Path("vllm_ascend/_310p/ops/minimax_m3.py"))
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--routes", type=int, default=MAX_ROUTES)
    parser.add_argument("--layout", choices=("canonical", "nz128", "nz256"), default="canonical")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        prepare(args.model, args.cache, args.bundle, args.layer, args.expert)
        return
    report = run(args.bundle, args.ops, args.binding, args.device, args.routes, args.layout)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["numerical_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
