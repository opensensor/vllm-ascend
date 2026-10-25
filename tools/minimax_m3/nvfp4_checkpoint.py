# SPDX-License-Identifier: Apache-2.0
"""Load source-preserving expert overrides for controlled full-model trials.

Dense weights still come from the audited FP16 export. Source activation scales
are retained in the bundle, but this path deliberately computes with FP16 inputs.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3.checkpoint import LAYER_NAME
from tools.minimax_m3.export_nvfp4 import MANIFEST, PROJECTIONS
from tools.minimax_m3.export_w4 import EXPERT_NAME, file_digest, local_file
from tools.minimax_m3.nvfp4 import LOOKUP_SIZE, NVFP4_LAYOUT
from tools.minimax_m3.nvfp4_native import GroupedW4Kernel
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts, SourceProjector


def read_source_manifest(folder, dense_manifest):
    result = json.loads((folder / MANIFEST).read_text())
    if result.get("complete") is not True or result.get("layout") != NVFP4_LAYOUT:
        raise ValueError("Incomplete or unsupported source NVFP4 bundle")
    for key in ("config_sha256", "index_sha256"):
        if result.get(key) != dense_manifest[key]:
            raise ValueError("Source NVFP4 and dense checkpoint provenance mismatch")
    ep_size = result.get("ep_size")
    if type(ep_size) is not int or ep_size <= 0:
        raise ValueError("Invalid source NVFP4 EP size")
    source_ids = set()
    for name, record in dense_manifest["weights"].items():
        match = EXPERT_NAME.fullmatch(name)
        if record["expert"] and match:
            source_ids.add((int(match[1]), int(match[2]), match[3]))
    exported_ids = set()
    for key, record in result["banks"].items():
        layer, rank, projection = record["layer"], record["ep_rank"], record["projection"]
        experts = record["experts"]
        if (
            key != f"layer-{layer:02d}-ep-{rank}-{projection}"
            or projection not in PROJECTIONS
            or not 0 <= rank < ep_size
            or type(experts) is not int
            or experts <= 0
        ):
            raise ValueError("Invalid source NVFP4 bank identity")
        local_file(folder, record["file"])
        exported_ids.update((layer, rank * experts + e, projection) for e in range(experts))
    if not source_ids or source_ids != exported_ids:
        raise ValueError("Source NVFP4 bundle does not cover the dense checkpoint's expert set")
    return result


def source_bank(folder, record):
    path = local_file(folder, record["file"])
    size, digest = path.stat().st_size, file_digest(path)
    if size != record["bytes"] or digest != record["sha256"]:
        raise ValueError(
            f"Source NVFP4 bank checksum mismatch: {path.name}; "
            f"size={size}/{record['bytes']}, sha256={digest}/{record['sha256']}"
        )
    experts, (rows, cols) = record["experts"], record["shape"]
    expected = {
        "codes": ([experts, rows, cols // 2], "I8"),
        "scales": ([experts, rows, cols // 16], "U8"),
        "lookup": ([experts, LOOKUP_SIZE], "F16"),
        "weight_global_scale": ([experts], "F32"),
        "input_global_scale": ([experts], "F32"),
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("Source NVFP4 bank tensor keys mismatch")
        for name, (shape, dtype) in expected.items():
            view = handle.get_slice(name)
            if view.get_shape() != shape or view.get_dtype() != dtype:
                raise ValueError("Source NVFP4 bank tensor metadata mismatch")
        # Never register file-backed, potentially unaligned pages as DMA input.
        # Keep one owned host tensor alive until its device transfer completes.
        for name in ("codes", "scales", "lookup"):
            yield name, handle.get_tensor(name).clone()


def load_source_banks(model, folder, dense_manifest, native_path, max_grouped_routes):
    manifest = read_source_manifest(folder, dense_manifest)
    native = NativeManifest(json.loads(native_path.read_text()))
    native.verify_files()
    assets = {Path(item["path"]).name: Path(item["path"]) for item in native.value["assets"]}
    if json.loads(assets["build-options.json"].read_text()).get("source_nvfp4") is not True:
        raise ValueError("Source NVFP4 requires a source-preserving native kernel")
    for library in native.value["libraries"]:
        torch.ops.load_library(library["path"])
    kernel = GroupedW4Kernel(native.name, str(assets["grouped_w4.bin"]), native.name + "_kernel")
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, PackedM3Experts):
                continue
            match = LAYER_NAME.search(module.prefix)
            if match is None or module.local_expert_offset % module.num_local_experts:
                raise ValueError("Source NVFP4 requires layer identity and contiguous EP banks")
            layer, rank = int(match[1]), module.local_expert_offset // module.num_local_experts
            lookups = {}
            for projection in PROJECTIONS:
                record = manifest["banks"][f"layer-{layer:02d}-ep-{rank}-{projection}"]
                target = getattr(module, projection + "_codes")
                expected = (record["experts"], record["shape"][0], record["shape"][1] // 2)
                if tuple(target.shape) != expected:
                    raise ValueError("Source NVFP4 model bank geometry mismatch")
                device = target.device
                for kind, tensor in source_bank(folder, record):
                    name = projection + "_" + kind
                    if kind == "codes":
                        setattr(module, name, target.view(torch.int8))
                        getattr(module, name).copy_(tensor)
                    elif kind == "scales":
                        setattr(module, name, tensor.to(device))
                    else:
                        module.register_buffer(name, tensor.to(device))
                    if device.type == "npu":
                        # Load-time only: retire DMA before releasing staging.
                        torch.npu.synchronize()
                lookups[getattr(module, projection + "_codes").data_ptr()] = getattr(module, projection + "_lookup")
                count += 1
            module.source_nvfp4 = True
            module.max_grouped_routes = max_grouped_routes
            module.projector = SourceProjector(kernel, lookups)
    if not count:
        raise ValueError("No source NVFP4 expert banks loaded")
    return count
