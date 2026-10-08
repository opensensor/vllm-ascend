# SPDX-License-Identifier: Apache-2.0
"""CPU-only scale export from permanent native checkpoints; no device work."""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .fused_weight_layout import FP16_SCALE_LAYOUT, PREROUNDED_SCALE_LAYOUT, tensor_digest
from .native_checkpoint import (
    CODE_NAME,
    INDEX,
    LAYOUT,
    MANIFEST,
    copy_kernel_bundle,
    file_digest,
    kernel_assets,
    source_geometry,
    validate_scale_contract,
    write_json,
)


def round_scales(value, fp16_storage=False):
    """Match half-round/promote arithmetic with FP32 or compact FP16 storage."""
    allowed_dtypes = (torch.float32, torch.float16) if fp16_storage else (torch.float32,)
    if value.device.type != "cpu" or value.dtype not in allowed_dtypes or value.ndim != 2:
        raise ValueError("scale export requires CPU FP32 block-scale matrices")
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite source weight scale")
    result = value.half().contiguous() if fp16_storage else value.half().float().contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("weight scale overflows the existing FP16 rounding contract")
    return result


def export(source, output, bundle, fp16_storage=False):
    """Link unchanged bytes and rewrite only small per-rank/layer scale shards.

    The complete marker is the final write. Partial output stays inspectable but
    is rejected by the model loader; existing destinations are never overwritten.
    No code unpacking, repacking, NPU import or serving action takes place.
    """
    if type(fp16_storage) is not bool:
        raise ValueError("FP16 scale storage selection must be boolean")
    marker = FP16_SCALE_LAYOUT if fp16_storage else PREROUNDED_SCALE_LAYOUT
    source, output, bundle = (Path(path).resolve() for path in (source, output, bundle))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / MANIFEST).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True or manifest.get("layout") != LAYOUT:
        raise ValueError("scale export requires a complete permanent native checkpoint")
    config, index, layers, experts = source_geometry(source)
    world = manifest["world_size"]
    if type(world) is not int or world <= 0 or experts % world:
        raise ValueError("invalid permanent expert partition")
    provenance, _ = kernel_assets(bundle)
    options = provenance["_build"]
    expected_option = "fp16_weight_scales" if fp16_storage else "prerounded_weight_scales"
    if options.get(expected_option) is not True:
        raise ValueError("scale export requires the matching scale storage kernel bundle")
    # Validate destination ABI before creating files. Coverage is verified below.
    other_option = "prerounded_weight_scales" if fp16_storage else "fp16_weight_scales"
    if options.get(other_option, False) is not False:
        raise ValueError("scale storage kernel options conflict")
    source_marker = manifest.get("weight_scale_layout")
    validate_scale_contract(
        manifest,
        config.get("ascend_glm_expert_scale_layout"),
        {"fp16_weight_scales": source_marker == FP16_SCALE_LAYOUT},
    )
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("unchanged native shards must share the source filesystem")
    for key in ("kernel_bundle", "indexer_kernel_bundle"):
        if key in manifest:
            relative = Path(manifest[key])
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("checkpoint bundle must be a relative child path")
    names = {name.removesuffix("_codes") + "_scale" for name in index["weight_map"] if CODE_NAME.fullmatch(name)}
    if not names <= index["weight_map"].keys():
        raise ValueError("permanent checkpoint lacks expert scales")
    partitions = defaultdict(list)
    for name in sorted(names):
        match = CODE_NAME.fullmatch(name.removesuffix("_scale") + "_codes")
        partitions[int(match[1]), int(match[2]) // (experts // world)].append(name)
    output.mkdir()
    # Publish an incomplete marker first, including failures during file copies.
    target_manifest = dict(
        manifest,
        complete=False,
        source=str(source),
        source_index_sha256=file_digest(source / INDEX),
        parent_native_manifest_sha256=file_digest(source / MANIFEST),
        weight_scale_layout=marker,
        scale_shards=[],
    )
    write_json(output / MANIFEST, target_manifest)
    for filename in sorted(set(index["weight_map"].values())):
        os.link(source / filename, output / filename)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (INDEX, MANIFEST, "config.json"):
            shutil.copy2(path, output / path.name)
    for key in ("indexer_kernel_bundle",):
        if key in manifest:
            relative = Path(manifest[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("checkpoint bundle must be a relative child path")
            shutil.copytree(source / relative, output / relative)
    copy_kernel_bundle(bundle, output / manifest["kernel_bundle"])
    records = []
    for (layer, rank), selected in sorted(partitions.items()):
        tensors, input_hashes = {}, {}
        shards = defaultdict(list)
        for name in selected:
            shards[index["weight_map"][name]].append(name)
        for shard, keys in sorted(shards.items()):
            with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
                for name in keys:
                    value = handle.get_tensor(name)
                    text = config.get("text_config", config)
                    hidden, intermediate = text["hidden_size"], text["moe_intermediate_size"]
                    expected_shape = (
                        (hidden // 32, intermediate // 32)
                        if name.endswith("down_proj_scale")
                        else (intermediate // 32, hidden // 32)
                    )
                    if value.shape != expected_shape:
                        raise ValueError("weight scale shape differs from the checkpoint configuration")
                    input_hashes[name] = tensor_digest(value)
                    if value.dtype == torch.float16 and source_marker != FP16_SCALE_LAYOUT:
                        raise ValueError("source FP16 scales lack the storage marker")
                    tensors[name] = round_scales(value, True) if fp16_storage else round_scales(value)
        filename = f"rounded-scales-rank{rank}-layer{layer:03d}.safetensors"
        path = output / filename
        temporary = path.with_suffix(".tmp")
        save_file(tensors, str(temporary), metadata={"layout": LAYOUT, "weight_scale_layout": marker})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name, expected in tensors.items():
                if not torch.equal(handle.get_tensor(name).view(torch.uint8), expected.view(torch.uint8)):
                    raise ValueError("scale shard readback changed bits")
                index["weight_map"][name] = filename
        records.append(
            dict(
                file=filename,
                layer=layer,
                rank=rank,
                bytes=path.stat().st_size,
                sha256=file_digest(path),
                tensor_count=len(tensors),
                tensor_bytes=sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
                source_tensor_sha256=input_hashes,
            )
        )
    if len(records) != len(layers) * world or sum(row["tensor_count"] for row in records) != len(names):
        raise ValueError("scale export did not cover every routed layer and partition")
    config["ascend_glm_expert_scale_layout"] = marker
    index["metadata"]["expert_scale_layout"] = marker
    write_json(output / "config.json", config)
    write_json(output / INDEX, index)
    target_manifest.update(
        complete=True,
        scale_shards=records,
        scale_tensor_count=len(names),
        kernel_provenance_sha256=file_digest(output / manifest["kernel_bundle"] / "provenance.json"),
        scale_export_device="cpu",
        scale_storage_dtype="float16" if fp16_storage else "float32",
        scale_compute_dtype="float32",
        scale_tensor_bytes=sum(row["tensor_bytes"] for row in records),
        transformed_code_tensors=0,
        hardware_validation="not_run",
        quality_validation="not_run",
    )
    write_json(output / MANIFEST, target_manifest)
    return target_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "kernel-bundle"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--fp16-storage", action="store_true", help="persist FP16 scales for matching FP32-compute kernels"
    )
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.kernel_bundle, fp16_storage=args.fp16_storage), indent=2))


if __name__ == "__main__":
    main()
