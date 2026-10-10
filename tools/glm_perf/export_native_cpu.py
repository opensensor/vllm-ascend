# SPDX-License-Identifier: Apache-2.0
"""Create a self-contained native checkpoint on CPU, preserving W2/W3/W4."""

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from .cpu_weight_layout import pack_canonical_cpu
from .deployment_profiles import profile, validate_profile
from .fused_weight_layout import FP16_SCALE_LAYOUT, MAX_PREPARE_THREADS
from .indexer_bundle import verified_manifest
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
from .prerounded_scale_checkpoint import round_scales

DENSE_SHARD_BYTES = 1024**3


def export(source, output, bundle, *, world_size=4, dense_shard_bytes=DENSE_SHARD_BYTES, cards=None, resume=False):
    """Pack canonical codes once; never widen/re-quantize existing W2 values.

    The output owns only indexed tensors: superseded source codes are not
    embedded in copied dense shards. An incomplete manifest remains on failure.
    Peak retained tensors are one expert partition or bounded dense shard.
    """
    selected_profile = profile(cards) if cards is not None else None
    if selected_profile is not None and world_size != selected_profile.expert_partition_size:
        raise ValueError("checkpoint expert partition does not match card profile")
    source, output, bundle = (Path(p).resolve() for p in (source, output, bundle))
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not output.is_dir():
        raise ValueError("resume requires an existing incomplete candidate")
    config, index, layers, experts = source_geometry(source)
    if (
        type(world_size) is not int
        or world_size < 1
        or experts % world_size
        or type(dense_shard_bytes) is not int
        or dense_shard_bytes < 1
    ):
        raise ValueError("invalid expert partition or dense shard budget")
    if config.get("ascend_glm_expert_layout") == LAYOUT:
        raise ValueError("CPU exporter requires canonical codes, not an already-native checkpoint")
    if config.get("ascend_glm_nz_packed_codes"):
        raise ValueError("CPU exporter requires canonical row-order codes, not an NZ overlay")
    provenance, _ = kernel_assets(bundle)
    options = provenance["_build"]
    if options.get("fp16_weight_scales") is not True or options.get("prerounded_weight_scales", False):
        raise ValueError("CPU export requires an FP16-scale native kernel bundle")
    weights = index["weight_map"]
    text = config.get("text_config", config)
    layer_types = text.get("mlp_layer_types")
    if layer_types is not None:
        if len(layer_types) != text["num_hidden_layers"]:
            raise ValueError("checkpoint MLP layer geometry is inconsistent")
        required_layers = {i for i, kind in enumerate(layer_types) if kind == "sparse"}
        required_layers.update(range(len(layer_types), len(layer_types) + text.get("num_nextn_predict_layers", 0)))
        if set(layers) != required_layers:
            raise ValueError("canonical checkpoint omits an entire routed or draft bank")
    hidden, intermediate = text["hidden_size"], text["moe_intermediate_size"]
    scale_names = {n.removesuffix("_codes") + "_scale" for n in weights if CODE_NAME.fullmatch(n)}
    if not scale_names <= weights.keys():
        raise ValueError("canonical checkpoint lacks routed expert scales")
    manifest = dict(
        schema_version=1,
        complete=False,
        layout=LAYOUT,
        world_size=world_size,
        num_experts=experts,
        layers=layers,
        activation_bits=4,
        weight_scale_layout=FP16_SCALE_LAYOUT,
        kernel_bundle="native-kernels",
        source_index_sha256=file_digest(source / INDEX),
        hardware_validation="not_run",
        quality_validation="not_run",
        dense_shard_bytes=dense_shard_bytes,
    )
    if resume:
        existing = json.loads((output / MANIFEST).read_text())
        if (
            existing.get("complete") is not False
            or any(
                existing.get(key) != manifest[key]
                for key in (
                    "schema_version",
                    "layout",
                    "world_size",
                    "num_experts",
                    "layers",
                    "activation_bits",
                    "weight_scale_layout",
                    "source_index_sha256",
                    "kernel_bundle",
                )
            )
            or existing.get("dense_shard_bytes", DENSE_SHARD_BYTES) != dense_shard_bytes
        ):
            raise ValueError("resume contract differs or candidate is complete")
        existing_provenance, _ = kernel_assets(output / manifest["kernel_bundle"])
        if existing_provenance != provenance:
            raise ValueError("resume kernel bundle differs")
    else:
        output.mkdir(parents=True)
        copy_kernel_bundle(bundle, output / manifest["kernel_bundle"])
    write_json(output / MANIFEST, manifest)
    weight_map, native_records, scale_records, dense_records, widths = {}, [], [], [], {}

    def save(filename, tensors, metadata):
        path = output / filename

        def matches(candidate):
            try:
                with safe_open(str(candidate), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != tensors.keys() or (handle.metadata() or {}) != metadata:
                        return False
                    for name, expected in tensors.items():
                        actual = handle.get_tensor(name)
                        if (
                            actual.dtype != expected.dtype
                            or actual.shape != expected.shape
                            or not torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
                        ):
                            return False
                return True
            except SafetensorError:
                return False

        if not (resume and path.exists() and matches(path)):
            temporary = path.with_suffix(path.suffix + ".tmp")
            save_file(tensors, str(temporary), metadata=metadata)
            temporary.replace(path)
        # Readback verifies identities/shape/dtype and bits before admission.
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != tensors.keys():
                raise ValueError("exported shard has different tensor identities")
            for name, expected in tensors.items():
                actual = handle.get_tensor(name)
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
                ):
                    raise ValueError("exported tensor readback differs")
                weight_map[name] = filename
        return dict(file=filename, bytes=path.stat().st_size, sha256=file_digest(path))

    per_rank = experts // world_size
    for layer in layers:
        for rank in range(world_size):
            tensors = {}
            selected = []
            for expert in range(rank * per_rank, (rank + 1) * per_rank):
                for projection in ("gate", "up", "down"):
                    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
                    selected.extend((stem + "_codes", stem + "_scale"))
            by_shard = defaultdict(list)
            for name in selected:
                by_shard[weights[name]].append(name)
            with ThreadPoolExecutor(max_workers=MAX_PREPARE_THREADS) as pool:
                pending = {}
                for shard, names in sorted(by_shard.items()):
                    with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
                        for name in names:
                            value = handle.get_tensor(name)
                            n, k = (hidden, intermediate) if ".down_proj_" in name else (intermediate, hidden)
                            if name.endswith("_codes"):
                                bits = value.shape[-1] * 8 // k
                                if value.ndim != 2 or value.shape != (n, k * bits // 8) or bits not in (2, 3, 4):
                                    raise ValueError("canonical code shape or precision is invalid")
                                previous = widths.setdefault(layer, bits)
                                if previous != bits:
                                    raise ValueError("native bank requires one width per layer")
                                pending[name] = pool.submit(pack_canonical_cpu, value, k)
                                if len(pending) >= MAX_PREPARE_THREADS * 2:
                                    first = next(iter(pending))
                                    tensors[first] = pending.pop(first).result()
                            else:
                                if value.shape != (n // 32, k // 32):
                                    raise ValueError("canonical scale shape differs from block contract")
                                tensors[name] = round_scales(value, True)
                for name, future in pending.items():
                    tensors[name] = future.result()
            record = save(
                f"native-rank{rank}-layer{layer:03d}.safetensors",
                tensors,
                {"layout": LAYOUT, "weight_scale_layout": FP16_SCALE_LAYOUT},
            )
            native_records.append(dict(record, layer=layer, rank=rank, tensor_count=3 * per_rank))
            scale_records.append(dict(record, layer=layer, rank=rank, tensor_count=3 * per_rank))
            del tensors

    dense, retained_bytes = {}, 0
    by_shard = defaultdict(list)
    for name, filename in weights.items():
        if not CODE_NAME.fullmatch(name) and name not in scale_names:
            by_shard[filename].append(name)
    for filename, names in sorted(by_shard.items()):
        with safe_open(str(source / filename), framework="pt", device="cpu") as handle:
            for name in sorted(names):
                value = handle.get_tensor(name)
                size = value.numel() * value.element_size()
                if dense and retained_bytes + size > dense_shard_bytes:
                    dense_records.append(save(f"dense-{len(dense_records):05d}.safetensors", dense, {}))
                    dense, retained_bytes = {}, 0
                # Own CPU storage so mapping lifetimes do not grow with readers.
                dense[name] = value.clone()
                retained_bytes += size
    if dense:
        dense_records.append(save(f"dense-{len(dense_records):05d}.safetensors", dense, {}))
    if set(weight_map) != set(weights):
        raise ValueError("native export lost tensor coverage")
    for path in source.iterdir():
        if path.is_file() and path.name not in (
            "config.json",
            INDEX,
            MANIFEST,
            "manifest.json",
            "progress.json",
            "README.md",
        ):
            if path.suffix != ".safetensors" and not path.name.startswith("native-export-"):
                shutil.copy2(path, output / path.name)
    config.update(
        ascend_glm_expert_layout=LAYOUT,
        ascend_glm_expert_scale_layout=FP16_SCALE_LAYOUT,
        ascend_glm_native_activation_bits=4,
        architectures=["Glm5NextW2ForCausalLM"],
    )
    write_json(output / "config.json", config)
    write_json(
        output / INDEX,
        {
            "metadata": {
                "total_size": sum(r["bytes"] for r in native_records + dense_records),
                "expert_scale_layout": FP16_SCALE_LAYOUT,
            },
            "weight_map": weight_map,
        },
    )
    manifest.update(
        native_shards=native_records,
        scale_shards=scale_records,
        scale_tensor_count=len(scale_names),
        dense_shards=dense_records,
        native_tensor_count=len(scale_names),
        layer_bits=widths,
        kernel_provenance_sha256=file_digest(output / "native-kernels/provenance.json"),
    )
    validate_scale_contract(manifest, FP16_SCALE_LAYOUT, options)
    (output / "README.md").write_text(
        "---\nlibrary_name: vllm\npipeline_tag: text-generation\ntags:\n- ascend\n- glm\n---\n\n"
        "# GLM native Ascend checkpoint candidate\n\n"
        "Requires the matching opensensor/vllm-ascend runtime and Ascend 310P kernel bundle. "
        "This is a hardware-specific native layout, not a generic CUDA quantization format.\n\n"
        f"Expert partition size: {world_size}; activation codes: A4; scale storage: FP16.\n\n"
        f"Per-layer weight widths: `{json.dumps(widths, sort_keys=True)}`.\n\n"
        "CPU export verified tensor coverage and readback. Hardware inference, accuracy, "
        "long-context capacity and performance have not been qualified for this artifact. "
        "Do not advertise it as validated until those gates pass.\n\n"
        "Quantization values were preserved from the canonical source; changing W2 to W3 "
        "requires a separate conversion from higher-precision weights. "
        "Consult the original model's license before redistribution.\n"
    )
    if selected_profile is not None:
        validate_profile(
            selected_profile.name,
            tensor_parallel_size=selected_profile.tensor_parallel_size,
            pipeline_parallel_size=selected_profile.pipeline_parallel_size,
            mtp_tokens=selected_profile.mtp_tokens,
            layer_bits=widths,
        )
        config["ascend_glm_deployment_profile"] = selected_profile.name
        manifest["deployment_profile"] = selected_profile.name
        write_json(output / "config.json", config)
    manifest["complete"] = True
    write_json(output / MANIFEST, manifest)
    return manifest


def attach_decode_bundle(model_path, bundle):
    """Add verified decode assets to an exported candidate without touching weights."""
    model_path, bundle = Path(model_path).resolve(), Path(bundle).resolve()
    manifest = json.loads((model_path / MANIFEST).read_text())
    if manifest.get("complete") is not True:
        raise ValueError("decode attachment requires a complete native checkpoint")
    if "decode_kernel_bundle" in manifest:
        raise ValueError("checkpoint already has a decode bundle")
    config = json.loads((model_path / "config.json").read_text())
    provenance, _ = kernel_assets(bundle)
    parent_options = json.loads((model_path / manifest["kernel_bundle"] / "provenance.json").read_text())["_build"]
    if any(provenance["_build"].get(key) == parent_options.get(key) for key in ("namespace", "helper_package")):
        raise ValueError("decode bundle requires a distinct operator namespace and helper package")
    validate_scale_contract(manifest, config.get("ascend_glm_expert_scale_layout"), provenance["_build"])
    name = "native-decode-kernels"
    destination = model_path / name
    if destination.exists():
        raise FileExistsError(destination)
    copy_kernel_bundle(bundle, destination)
    manifest.update(
        hardware_validation="not_run",
        quality_validation="not_run",
        decode_kernel_bundle=name,
        decode_kernel_provenance_sha256=file_digest(destination / "provenance.json"),
    )
    write_json(model_path / MANIFEST, manifest)
    return manifest


def attach_indexer_bundle(model_path, bundle):
    """Preserve the prepared AI-Core indexer with relocatable, hashed assets."""
    model_path, bundle = Path(model_path).resolve(), Path(bundle).resolve()
    manifest = json.loads((model_path / MANIFEST).read_text())
    if manifest.get("complete") is not True or "indexer_kernel_bundle" in manifest:
        raise ValueError("indexer attachment requires a complete checkpoint without an indexer bundle")
    source, _ = verified_manifest(bundle)
    destination = model_path / "indexer-kernels"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    portable = source.value
    for field in ("libraries", "assets"):
        for entry in portable[field]:
            path = Path(entry["path"])
            relative = path.relative_to(bundle)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            entry["path"] = str(relative)
    write_json(destination / "manifest.json", portable)
    verified_manifest(destination)
    manifest.update(
        indexer_kernel_bundle=destination.name,
        indexer_manifest_sha256=file_digest(destination / "manifest.json"),
        hardware_validation="not_run",
        quality_validation="not_run",
    )
    write_json(model_path / MANIFEST, manifest)
    return manifest


def attach_attribution(model_path, original_source):
    """Copy original license/attribution bytes for legacy canonical inputs."""
    model_path, original_source = Path(model_path).resolve(), Path(original_source).resolve()
    manifest = json.loads((model_path / MANIFEST).read_text())
    if manifest.get("complete") is not True:
        raise ValueError("attribution attachment requires a complete candidate")
    original_config = json.loads((original_source / "config.json").read_text())
    candidate_config = json.loads((model_path / "config.json").read_text())
    original_text = original_config.get("text_config", original_config)
    candidate_text = candidate_config.get("text_config", candidate_config)
    if any(
        original_text.get(key) != candidate_text.get(key)
        for key in ("num_hidden_layers", "hidden_size", "moe_intermediate_size", "n_routed_experts")
    ):
        raise ValueError("attribution source model geometry differs")
    licenses = [
        p for p in original_source.iterdir() if p.is_file() and p.name.upper().startswith(("LICENSE", "NOTICE"))
    ]
    if not licenses:
        raise FileNotFoundError("original source has no license or notice file")
    files = [(path, path.name) for path in licenses]
    if (original_source / "README.md").is_file():
        files.append((original_source / "README.md", "SOURCE_README.md"))
    for source, name in files:
        target = model_path / name
        if target.exists() and file_digest(target) != file_digest(source):
            raise ValueError("existing attribution differs from original source")
    records = []
    for source, name in files:
        target = model_path / name
        if not target.exists():
            shutil.copy2(source, target)
        records.append(dict(file=name, sha256=file_digest(target)))
    manifest["source_attribution"] = dict(
        files=records, original_config_sha256=file_digest(original_source / "config.json")
    )
    write_json(model_path / MANIFEST, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "bundle"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--cards", type=int, choices=(2, 3), help="Bind precision and runtime topology for release")
    parser.add_argument("--decode-bundle", help="Optional separately qualified decode kernel bundle")
    parser.add_argument("--indexer-bundle", help="Optional prepared AI-Core indexer bundle")
    parser.add_argument("--attribution-source", help="Original model directory containing license and model card")
    parser.add_argument(
        "--resume", action="store_true", help="Verify and reuse identical shards of an incomplete candidate"
    )
    args = parser.parse_args()
    manifest = export(
        args.source, args.output, args.bundle, world_size=args.world_size, cards=args.cards, resume=args.resume
    )
    if args.decode_bundle:
        manifest = attach_decode_bundle(args.output, args.decode_bundle)
    if args.indexer_bundle:
        manifest = attach_indexer_bundle(args.output, args.indexer_bundle)
    if args.attribution_source:
        manifest = attach_attribution(args.output, args.attribution_source)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
