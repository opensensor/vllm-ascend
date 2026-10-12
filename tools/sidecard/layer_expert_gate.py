# SPDX-License-Identifier: Apache-2.0
"""Clipped routed arithmetic against the frozen independent CPU quantizer."""

import argparse
import importlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from .device_backend import CudaOracle
from .layer_weights import TOP_K, WIDTH, CudaExperts, NpuExperts, PlanBanks, private_native_banks
from .native_expert_probe import contained, digest_file
from .resident import NativeResidentPlan


def run(checkpoint, specification, device, output):
    import torch

    from tools.glm_perf.fused_moe_profile import frozen_helper

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if device.startswith("npu"):
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
            raise ValueError("clipped expert gate must preserve the other Ascend card")
        import torch_npu

        torch.npu.set_device(int(device.split(":")[1]))
        torch_npu.npu.set_compile_mode(jit_compile=False)
    specification = json.loads(specification.read_bytes())
    manifest = json.loads((checkpoint / "native-layout.json").read_bytes())
    bundle = contained(checkpoint, manifest["kernel_bundle"])
    helper, options = frozen_helper(bundle) if device.startswith("npu") else (None, manifest)
    # CUDA Python cannot load the NPU bridge. The CPU reference is copied from
    # the authenticated frozen package, whose operators need Torch only.
    package = json.loads((bundle / "provenance.json").read_bytes())["_build"]["helper_package"]
    if helper is None:
        for name, checksum in json.loads((bundle / "provenance.json").read_bytes())["_helpers"].items():
            if digest_file(bundle / package / name) != checksum:
                raise ValueError("CPU reference helper authentication failed")
        sys.path.insert(0, str(bundle))
    projection = importlib.import_module(package + ".fused_moe_probe").projection_reference
    unpack = importlib.import_module(package + ".fused_weight_layout").unpack_cube
    cases = []
    layers = (18, 32) if device.startswith("npu") else (3, 8, 33)
    for layer in layers:
        plan = NativeResidentPlan(
            checkpoint, tuple((layer, expert) for expert in range(TOP_K)), specification["checkpoint_sha256"]
        )
        plan.authenticate()
        bank = next(private_native_banks(plan, torch) if device.startswith("npu") else plan.items())[1]
        gate, gs, down, ds = bank["tensors"]
        signed_gate, signed_down = unpack(gate, WIDTH), unpack(down, 2048)
        if device.startswith("npu"):
            override_path = Path(specification["native_gate_override"])
            override = json.loads(override_path.read_bytes())

            def factory(path, entry, override=override, override_path=override_path):
                filename = Path(path).name
                if filename in override["native_gate_files"]:
                    path = str(override_path.parent / filename)
                    if digest_file(Path(path)) != override["native_gate_files"][filename]:
                        raise ValueError("clipped native gate checksum differs")
                return getattr(torch.classes, options["namespace"]).Kernel(path, entry)

            actor = object.__new__(NpuExperts)
            actor.torch, actor.rank, actor.reduce_results = torch, 0, False
            actor.stable_rows = helper.FUSED_REDUCTION_TOKENS + 1
            actor.native = helper.NativeFusedMoE(
                bundle,
                namespace=options["namespace"],
                activation_bits=4,
                kernel_factory=factory,
                prepared_weight_layout=True,
                fp16_route_workspace=True,
            )
            actor.banks = {layer: tuple(value.clone().to(device) for value in bank["tensors"])}
        else:
            actor = object.__new__(CudaExperts)
            actor.torch = torch
            actor.oracle = CudaOracle(PlanBanks(plan), specification["cuda_backend"], TOP_K * 32)
            actor.configure_stream(specification["cuda_backend"].get("torch_stream_experts", False))
        try:
            for rows in (1, 4, 17):
                for phase, multiplier in enumerate((1.0, 4.0)):
                    torch.manual_seed(91373 + rows + layer)
                    hidden = (torch.randn(rows, WIDTH) * multiplier).half()
                    ids = torch.arange(TOP_K)[None, :].expand(rows, -1).contiguous()
                    weights = (torch.arange(1, TOP_K + 1).float() / 36 * 2.5)[None, :].expand(rows, -1).contiguous()
                    weights[-1, -1] = 0
                    expected = torch.zeros(rows, WIDTH)
                    clipped_values = 0
                    for expert in range(TOP_K):
                        gu = projection(hidden, signed_gate[expert], gs[expert].float(), 4)
                        g, u = gu.float().chunk(2, -1)
                        clipped_values += int(((g > 10) | (u.abs() > 10)).sum())
                        activation = (torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10)).half()
                        projected = projection(activation, signed_down[expert], ds[expert].float(), 4)
                        expected += projected.float() * weights[:, expert, None]
                    stream = torch.cuda.Stream() if device == "cuda" and getattr(actor, "use_stream", False) else None
                    with torch.cuda.stream(stream) if stream is not None else nullcontext():
                        actual = actor(layer, hidden.to(device), ids.to(device), weights.to(device)).cpu()
                    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
                    with torch.cuda.stream(stream) if stream is not None else nullcontext():
                        repeated = actor(layer, hidden.to(device), ids.to(device), weights.to(device)).cpu()
                    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
                    cases.append(
                        dict(
                            layer=layer,
                            rows=rows,
                            phase=phase,
                            clipping_activated_elements=clipped_values,
                            max_abs_error=float((actual - expected).abs().max()),
                            passed=True,
                            repeat_equal=True,
                            nondefault_cuda_stream=stream is not None,
                        )
                    )
        finally:
            if not device.startswith("npu"):
                actor.close()
        del actor, bank, signed_gate, signed_down
    report = dict(passed=True, device=device, cases=cases, swiglu_limit=10.0, real_checkpoint_weights=True)
    if not any(case["clipping_activated_elements"] for case in cases):
        raise AssertionError("fixture did not exercise routed clipping")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "npu:0", "npu:1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkpoint, args.specification, args.device, args.output)
    print(json.dumps({"passed": result["passed"], "device": args.device}), flush=True)


if __name__ == "__main__":
    main()
