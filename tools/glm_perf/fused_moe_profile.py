# SPDX-License-Identifier: Apache-2.0
"""Cold device-event comparison of frozen fused-MoE builds on real experts."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

from .glm_int4 import pack_nz_codes, unpack_canonical_codes


def frozen_helper(build):
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    package = options["helper_package"]
    root = build / package
    for name, digest in provenance["_helpers"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen helper changed: " + name)
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[package].__file__).resolve() != (root / "__init__.py").resolve():
        raise ValueError("helper package belongs to another build")
    bridge = build / f"glm_reconstruction_bridge_v{options['version']}.so"
    if hashlib.sha256(bridge.read_bytes()).hexdigest() != provenance["reconstruction_bridge.cpp"]["binary_sha256"]:
        raise ValueError("bridge differs from provenance")
    for stage in ("pack", "gate_up", "down"):
        name = f"glm_fused_{stage}.bin"
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != provenance[name]["binary_sha256"]:
            raise ValueError("kernel differs from provenance: " + name)
    if "glm_fused_route_input.bin" in provenance:
        name = "glm_fused_route_input.bin"
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != provenance[name]["binary_sha256"]:
            raise ValueError("routed input kernel differs from provenance")
    if "glm_fused_reduce.bin" in provenance:
        name = "glm_fused_reduce.bin"
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != provenance[name]["binary_sha256"]:
            raise ValueError("kernel differs from provenance: " + name)
    torch.ops.load_library(str(bridge))
    return importlib.import_module(package + ".glm_fused_moe"), options


def measure(native, pipeline, warmups, samples):
    for _ in range(warmups):
        pipeline()
    torch.npu.synchronize()
    original = native.launch
    names = {
        id(native.pack_kernel): "input_quant_once",
        id(native.gate_kernel): "gate_up_swiglu_quant",
        id(native.down_kernel): "down_weighted_reduce",
    }
    for attribute, name in (("gate_w3_kernel", "gate_up_swiglu_quant"), ("down_w3_kernel", "down_weighted_reduce")):
        kernel = getattr(native, attribute, None)
        if kernel is not None:
            names[id(kernel)] = name
    if getattr(native, "route_input_kernel", None) is not None:
        names[id(native.route_input_kernel)] = "route_input_pack"
    if hasattr(native, "reduce_kernel"):
        names[id(native.reduce_kernel)] = "stable_prefill_reduce"
    # A bulk-prefill pipeline has four stages. Allocating two events per stage
    # for every sample can exhaust the 310P event pool. Reuse a bounded pair
    # per stage, resolving each sample before recording the next one.
    events = {key: (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for key in names}
    seen = set()
    values = {name: [] for name in names.values()}

    def timed(kernel, args, blocks):
        key = id(kernel)
        if any(names[previous] == names[key] for previous in seen):
            raise RuntimeError("event profiler requires one launch per fused stage per sample")
        start, end = events[key]
        start.record()
        original(kernel, args, blocks)
        end.record()
        seen.add(key)

    native.launch = timed
    try:
        for _ in range(samples):
            seen.clear()
            pipeline()
            torch.npu.synchronize()
            for key in seen:
                start, end = events[key]
                values[names[key]].append(start.elapsed_time(end))
    finally:
        native.launch = original
    return {name: {"median_ms": statistics.median(ms), "samples_ms": ms} for name, ms in values.items() if ms}


def run(builds, checkpoint, output, layers=(10, 11, 33), tokens=2, warmups=3, samples=9):
    # NPU-only dependencies stay out of CPU geometry and test imports.
    import torch_npu
    from safetensors import safe_open

    if output.exists():
        raise FileExistsError(output)
    if not builds or tokens < 1 or warmups < 1 or samples < 3:
        raise ValueError("require builds, positive tokens/warmups and at least three samples")
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    loaded = [(build.resolve(), *frozen_helper(build.resolve())) for build in builds]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    payload = {
        "complete": False,
        "scope": "one real expert per layer, synthetic activations, prepared topk1 routing; device launch intervals",
        "quality_evaluated": False,
        "tokens": tokens,
        "warmups": warmups,
        "samples": samples,
        "records": [],
    }
    try:
        for layer in layers:
            prefix = f"model.language_model.layers.{layer}.mlp.experts.0."
            projections = []
            hashes = {}
            for name in ("gate", "up", "down"):
                tensors = []
                for suffix in ("_codes", "_scale"):
                    key = prefix + name + "_proj" + suffix
                    with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as handle:
                        tensors.append(handle.get_tensor(key))
                codes, scales = tensors
                hashes[name] = hashlib.sha256(codes.numpy().tobytes() + scales.numpy().tobytes()).hexdigest()
                projections.append((unpack_canonical_codes(codes, scales.shape[1] * 32), scales.float()))
            gate = torch.cat([p[0] for p in projections[:2]], 0)[None]
            gs = torch.cat([p[1] for p in projections[:2]], 0)[None]
            down, ds = projections[2][0][None], projections[2][1][None]
            hidden, inter = gate.shape[-1], down.shape[-1]
            bits = codes.shape[-1] * 8 // inter
            x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(5704)).half().npu()
            weights = torch.ones(tokens, 1).npu()
            order = torch.arange(tokens, dtype=torch.int64).npu()
            ends = torch.tensor([tokens], dtype=torch.int64).npu()
            for build, helper, options in loaded:
                # Prepare fixture scales outside the measured pipeline. The
                # production path reads these exact FP32 values from disk.
                prepared = options.get("prerounded_weight_scales", False)
                ggs, gds = ((value.half().float() if prepared else value).npu() for value in (gs, ds))
                for activation_bits in (8, 4):
                    layout_options = {"prepared_weight_layout": True} if options.get("prepared_weight_layout") else {}
                    if options.get("weight_decode_lut"):
                        layout_options["weight_decode_lut"] = True
                    if options.get("fp16_route_workspace"):
                        layout_options["fp16_route_workspace"] = True
                    native = helper.NativeFusedMoE(
                        build, namespace=options["namespace"], activation_bits=activation_bits, **layout_options
                    )
                    pack = getattr(native, "pack_weight_codes", pack_nz_codes)
                    gc, dc = pack(gate, bits).npu(), pack(down, bits).npu()
                    geometry = helper.FusedGeometry(tokens, 1, 1, hidden, inter, bits, bits, activation_bits)

                    pipeline = partial(native.grouped, x, gc, ggs, dc, gds, weights, order, ends, geometry)
                    record = {
                        "build_options": options,
                        "provenance_sha256": hashlib.sha256((build / "provenance.json").read_bytes()).hexdigest(),
                        "tensor_prefix": prefix,
                        "checkpoint_tensor_hashes": hashes,
                        "weight_bits": bits,
                        "activation_bits": activation_bits,
                        "prerounded_weight_scales": prepared,
                        "stage_ms": measure(native, pipeline, warmups, samples),
                    }
                    payload["records"].append(record)
                    print(json.dumps(record), flush=True)
            del pipeline, x, gc, dc, ggs, gds, weights, order, ends, native
        payload["complete"] = True
    finally:
        output.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", action="append", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    run(args.build_dir, args.checkpoint, args.output, tokens=args.tokens, warmups=args.warmups, samples=args.samples)


if __name__ == "__main__":
    main()
