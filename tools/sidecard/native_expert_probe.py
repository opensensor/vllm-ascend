# SPDX-License-Identifier: Apache-2.0
"""Qualify single real native GLM experts with top-k=1 and unit route weights.

Each input row is one route, so the existing native reducer returns one raw
expert row. This is an arithmetic gate for a possible adapter, not a production
backend: compact route gathering, complete owner-bank loading and CUDA parity
are still required. Run in a dedicated Ascend process with the qualified
tools.glm_perf package available on its Python path.
"""

import argparse
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path

from .checkpoint import read_header
from .placement import validate_glm_config

NATIVE_LAYOUT = "cube_n128_k256_v1"
SCALE_LAYOUT = "fp16_storage_fp32_compute_v1"
MAX_PROBE_ROWS = 32
HASH_CHUNK_BYTES = 8 * 1024**2


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("native probe path escapes checkpoint")
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_expert(root, layer, expert):
    """Bounded header admission; payload authentication happens before upload."""
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_bytes())
    validate_glm_config(config)
    manifest = json.loads((root / "native-layout.json").read_bytes())
    if (
        manifest.get("complete") is not True
        or manifest.get("schema_version") != 1
        or manifest.get("layout") != NATIVE_LAYOUT
        or manifest.get("weight_scale_layout") != SCALE_LAYOUT
        or config.get("ascend_glm_expert_layout") != NATIVE_LAYOUT
        or config.get("ascend_glm_expert_scale_layout") != SCALE_LAYOUT
        or type(layer) is not int
        or not 3 <= layer < 45
        or type(expert) is not int
        or not 0 <= expert < 288
    ):
        raise ValueError("native probe requires a complete main-model FP16-scale Cube checkpoint")
    bits = manifest["layer_bits"].get(str(layer))
    if bits not in (2, 3, 4) or manifest.get("activation_bits") not in (4, 8):
        raise ValueError("unsupported native probe precision")
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    shard_records = {row["file"]: row for row in manifest["native_shards"]}
    scale_records = {row["file"]: row for row in manifest["scale_shards"]}
    tensors, files, headers = {}, {}, {}
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    for projection in ("gate", "up", "down"):
        rows, columns = (4096, 2048) if projection == "down" else (2048, 4096)
        for kind in ("codes", "scale"):
            name = prefix + projection + "_proj_" + kind
            filename = index[name]
            records = shard_records if kind == "codes" else scale_records
            if filename not in records:
                raise ValueError("selected tensor lacks a native shard receipt")
            record = records[filename]
            if filename in files and files[filename] != record["sha256"]:
                raise ValueError("native shard receipts disagree")
            files[filename] = record["sha256"]
            if filename not in headers:
                headers[filename] = read_header(contained(root, filename))[0]
            metadata = headers[filename].get("__metadata__", {})
            if metadata.get("layout") != NATIVE_LAYOUT or metadata.get("weight_scale_layout") != SCALE_LAYOUT:
                raise ValueError("native shard lacks matching layout metadata")
            descriptor = headers[filename][name]
            expected = (rows, columns * bits // 8) if kind == "codes" else (rows // 32, columns // 32)
            if tuple(descriptor["shape"]) != expected or descriptor["dtype"] not in (
                ("I8", "U8") if kind == "codes" else ("F16",)
            ):
                raise ValueError("selected native expert has incompatible shape or dtype")
            tensors[name] = filename
    return {
        "root": str(root),
        "layer": layer,
        "expert": expert,
        "bits": bits,
        "activation_bits": manifest["activation_bits"],
        "tensor_files": tensors,
        "shard_sha256": files,
        "manifest_sha256": digest_file(root / "native-layout.json"),
        "config_sha256": digest_file(root / "config.json"),
        "index_sha256": digest_file(root / "model.safetensors.index.json"),
        "kernel_bundle": manifest["kernel_bundle"],
        "kernel_provenance_sha256": manifest["kernel_provenance_sha256"],
        "transfer_hardware_validated": False,
        "expert_math_validated": False,
        "complete_owner_bank_loaded": False,
        "real_glm_inference_validated": False,
    }


def run(root, layers, expert, device_index, rows, samples):
    if (
        not layers
        or not rows
        or any(type(count) is not int or not 1 <= count <= MAX_PROBE_ROWS for count in rows)
        or type(device_index) is not int
        or not 0 <= device_index < 4
        or type(samples) is not int
        or not 1 <= samples <= 20
    ):
        raise ValueError("invalid native expert probe bounds")
    plans = [select_expert(root, layer, expert) for layer in layers]
    for plan in plans:
        for filename, expected in plan["shard_sha256"].items():
            if digest_file(contained(root, filename)) != expected:
                raise ValueError("native expert shard failed checksum verification")
    bundle = contained(root, plans[0]["kernel_bundle"])
    if digest_file(bundle / "provenance.json") != plans[0]["kernel_provenance_sha256"]:
        raise ValueError("native kernel provenance differs from checkpoint")

    # Device-only dependencies and dynamic native helpers stay in this explicit
    # accelerator gate. Never import them from the offline planning process.
    import torch
    import torch_npu
    from safetensors import safe_open

    from tools.glm_perf.fused_moe_probe import reference
    from tools.glm_perf.fused_moe_profile import frozen_helper

    torch.set_num_threads(1)
    torch.npu.set_device(device_index)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    helper, options = frozen_helper(bundle)
    layout_helper = importlib.import_module(helper.__package__ + ".fused_weight_layout")
    native = helper.NativeFusedMoE(
        bundle,
        namespace=options["namespace"],
        activation_bits=plans[0]["activation_bits"],
        prepared_weight_layout=True,
        **({"fp16_route_workspace": True} if options.get("fp16_route_workspace") else {}),
    )
    if native.weight_scale_dtype != torch.float16:
        raise ValueError("native expert probe requires permanent FP16 scale storage")
    report = {
        "scope": "one resident real expert per layer; synthetic FP16 activations; raw top-k=1/unit-weight output",
        "device": device_index,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "samples": samples,
        "complete_owner_bank_loaded": False,
        "cuda_expert_math_validated": False,
        "real_glm_inference_validated": False,
        "capacity_admitted": False,
        "plans": plans,
        "cases": [],
    }
    for plan in plans:
        tensors = {}
        for name, filename in plan["tensor_files"].items():
            # Clone before upload: do not pin mmap-backed safetensors pages.
            with safe_open(str(contained(root, filename)), framework="pt", device="cpu") as handle:
                tensors[name.rsplit(".", 1)[1]] = handle.get_tensor(name).clone()
        gate = torch.cat((tensors["gate_proj_codes"], tensors["up_proj_codes"]), 0)[None].contiguous()
        gs = torch.cat((tensors["gate_proj_scale"], tensors["up_proj_scale"]), 0)[None].contiguous()
        down, ds = tensors["down_proj_codes"][None], tensors["down_proj_scale"][None]
        signed_gate = layout_helper.unpack_cube(gate, 4096)
        signed_down = layout_helper.unpack_cube(down, 2048)
        banks = tuple(value.to(native.device).contiguous() for value in (gate, gs, down, ds))
        torch.npu.synchronize()
        generator = torch.Generator().manual_seed(7211 + plan["layer"])
        for count in rows:
            hidden = torch.randn(count, 4096, generator=generator).half()
            ids = torch.full((count, 1), expert, dtype=torch.int32)
            weights = torch.ones(count, 1, dtype=torch.float32)
            d_ids, d_weights = ids.to(native.device), weights.to(native.device)
            d_hidden = hidden.to(native.device)
            previous = None
            for pass_index, multiplier in enumerate((1.0, 0.75)):
                changed = (hidden * multiplier).half()
                d_hidden.copy_(changed.to(native.device))
                expected = reference(
                    changed,
                    signed_gate,
                    gs.float(),
                    signed_down,
                    ds.float(),
                    weights,
                    ids,
                    plan["activation_bits"],
                    offset=expert,
                )
                output = native(d_hidden, *banks, d_weights, d_ids, expert)
                torch.npu.synchronize()
                actual = output.cpu().clone()
                torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
                durations = []
                for _ in range(samples):
                    start = time.perf_counter()
                    output = native(d_hidden, *banks, d_weights, d_ids, expert)
                    torch.npu.synchronize()
                    durations.append((time.perf_counter() - start) * 1000)
                torch.testing.assert_close(output.cpu(), actual, rtol=0, atol=0)
                if previous is not None and torch.equal(actual, previous):
                    raise AssertionError("changed activations produced a stale expert result")
                previous = actual
                report["cases"].append(
                    {
                        "layer": plan["layer"],
                        "expert": expert,
                        "weight_bits": plan["bits"],
                        "rows": count,
                        "pass": pass_index,
                        "max_absolute_error": float((actual - expected).abs().max()),
                        "implementation_nrmse": float(
                            (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
                        ),
                        "median_synchronized_ms": statistics.median(durations),
                        "samples_synchronized_ms": durations,
                        "repeat_bitwise_equal": True,
                    }
                )
        del banks
        torch.npu.synchronize()
    report["expert_math_validated"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 8, 33])
    parser.add_argument("--expert", type=int, default=72)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 17])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = (
        [select_expert(args.checkpoint, layer, args.expert) for layer in args.layers]
        if args.plan_only
        else run(args.checkpoint, args.layers, args.expert, args.device, args.rows, args.samples)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
