# SPDX-License-Identifier: Apache-2.0
"""Permanent channel-scale expert overlay made directly from original FP8 weights.

The overlay supplements a base model; it is not a standalone Hugging Face model.
Preparation is CPU-only, bounded to one expert, and never reads a W2/W3 bank.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .channel_scale_layout import FACTOR_LAYOUT, NativeWeightBank, native_weight_parameters
from .channel_scale_math import LAYOUT, quantize_weights

SOURCE_PREFIX = "model.language_model.layers"
PROJECTIONS = ("gate", "up", "down")
FP8_BLOCK = 128


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read_fp8_projection(source, index, layer, expert, projection):
    """Clone mmap storage and dequantize the source's 128x128 FP8 blocks on CPU."""
    if projection not in PROJECTIONS or type(layer) is not int or type(expert) is not int:
        raise ValueError("invalid original expert projection")
    prefix = f"{SOURCE_PREFIX}.{layer}.mlp.experts.{expert}.{projection}_proj"
    result = []
    for suffix in (".weight", ".weight_scale_inv"):
        name = prefix + suffix
        filename = index.get(name)
        if not filename or Path(filename).name != filename:
            raise ValueError("original FP8 projection missing or unsafe shard path: " + name)
        with safe_open(str(Path(source) / filename), framework="pt", device="cpu") as handle:
            result.append(handle.get_tensor(name).clone())
    weight, scale = result
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or scale.dtype != torch.float32:
        raise ValueError("channel-scale export requires original E4M3 FP8 weights and FP32 scales")
    n, k = weight.shape
    if n % FP8_BLOCK or k % FP8_BLOCK or scale.shape != (n // FP8_BLOCK, k // FP8_BLOCK):
        raise ValueError("source FP8 block geometry differs")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("source FP8 scales must be positive finite values")
    value = weight.float().reshape(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    return (value * scale[:, None, :, None]).reshape(n, k).contiguous()


def save_expert(path, gate_up, down):
    gate_up.validate()
    down.validate()
    tensors = {}
    for name, bank in (("gate_up", gate_up), ("down", down)):
        for field, value in zip(("codes", "scales", "sums", "smoothing"), native_weight_parameters(bank)):
            tensors[name + "." + field] = value
    save_file(
        tensors,
        str(path),
        metadata={
            "layout": LAYOUT,
            "factor_layout": FACTOR_LAYOUT,
            "gate_group": str(gate_up.group_size),
            "down_group": str(down.group_size),
            "source": "original_fp8_block128",
            "qualification": "offline_candidate",
        },
    )


def load_expert(path, expected_hash):
    """Fail closed before reading tensors; no runtime repacking or scale conversion."""
    if file_sha256(path) != expected_hash:
        raise ValueError("channel-scale expert checksum differs")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if (
            metadata.get("layout") != LAYOUT
            or metadata.get("source") != "original_fp8_block128"
            or metadata.get("factor_layout") != FACTOR_LAYOUT
        ):
            raise ValueError("requires original-FP8 channel-scale expert overlay")
        banks = []
        for name, key in (("gate_up", "gate_group"), ("down", "down_group")):
            fields = [handle.get_tensor(name + "." + f).clone() for f in ("codes", "scales", "sums", "smoothing")]
            banks.append(NativeWeightBank(*fields, int(metadata[key])).validate())
    return tuple(banks)


def load_partition(root, layer, first, stop, *, host_budget_bytes):
    """Stage only the local expert range, with a budget check before CPU stacking."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("complete") is not True
        or manifest.get("layout") != LAYOUT
        or manifest.get("factor_layout") != FACTOR_LAYOUT
    ):
        raise ValueError("requires complete channel-scale overlay manifest")
    if any(type(v) is not int for v in (layer, first, stop, host_budget_bytes)) or not 0 <= first < stop:
        raise ValueError("invalid expert partition or host budget")
    entries = [manifest["experts"][f"{layer}:{expert}"] for expert in range(first, stop)]
    if any(Path(e["file"]).name != e["file"] for e in entries):
        raise ValueError("unsafe channel-scale expert path")
    if sum((root / e["file"]).stat().st_size for e in entries) * 2 > host_budget_bytes:
        raise ValueError("expert partition exceeds CPU staging budget")
    parts = [[], []]
    groups = None
    for entry in entries:
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("unsafe channel-scale expert path")
        banks = load_expert(root / filename, entry["sha256"])
        current = tuple(b.group_size for b in banks)
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for bank in banks
            for value in (bank.codes, bank.scales, bank.sums, bank.smoothing)
        )
        if (
            current != (manifest["gate_group"], manifest["down_group"])
            or list(banks[0].shape) != entry["gate_shape"]
            or list(banks[1].shape) != entry["down_shape"]
            or tensor_bytes != entry["tensor_bytes"]
        ):
            raise ValueError("expert tensors differ from admitted manifest geometry")
        if groups is not None and current != groups:
            raise ValueError("expert partition mixes numerical contracts")
        groups = current
        for parts_for_projection, bank in zip(parts, banks):
            parts_for_projection.append(bank)
    result = []
    for banks, group in zip(parts, groups):
        fields = [torch.cat([getattr(b, f) for b in banks], dim=0) for f in ("codes", "scales", "sums", "smoothing")]
        result.append(NativeWeightBank(*fields, group).validate())
    return tuple(result)


def write_manifest(path, manifest):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(pending, path)


def export(source, destination, layers, experts, gate_group, down_group, clipping=1.0, *, resume=False):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if not layers or not experts or len(set(layers)) != len(layers) or len(set(experts)) != len(experts):
        raise ValueError("export requires unique nonempty layer and expert selections")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [FP8_BLOCK, FP8_BLOCK]:
        raise ValueError("export requires original block128 FP8 source, never a low-bit derived checkpoint")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    destination.mkdir(parents=True, exist_ok=resume)
    manifest = {
        "layout": LAYOUT,
        "factor_layout": FACTOR_LAYOUT,
        "complete": False,
        "source": str(source),
        "source_index_sha256": file_sha256(index_path),
        "layers": layers,
        "expert_ids": experts,
        "gate_group": gate_group,
        "down_group": down_group,
        "clipping": clipping,
        "smoothing": "identity; real activation calibration not supplied",
        "qualification": "offline_candidate; no full-model accuracy or NPU latency validation",
        "experts": {},
    }
    manifest_path = destination / "manifest.json"
    journal_path = destination / "completed.jsonl"
    selected_keys = {f"{layer}:{expert}" for layer in layers for expert in experts}
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in manifest.items() if key not in ("complete", "experts")):
            raise ValueError("resume requires identical source, selections and numerical contract")
        if journal_path.exists():
            for line in journal_path.read_text().splitlines():
                record = json.loads(line)
                entry = record["entry"]
                if (
                    record["key"] not in selected_keys
                    or record["key"] in manifest["experts"]
                    or Path(entry["file"]).name != entry["file"]
                    or file_sha256(destination / entry["file"]) != entry["sha256"]
                ):
                    raise ValueError("completed expert differs from durable export journal")
                manifest["experts"][record["key"]] = entry
    write_manifest(manifest_path, manifest)
    for layer in layers:
        for expert in experts:
            key = f"{layer}:{expert}"
            if key in manifest["experts"]:
                continue
            gate = read_fp8_projection(source, index, layer, expert, "gate")
            up = read_fp8_projection(source, index, layer, expert, "up")
            gate_up = quantize_weights(torch.cat((gate, up))[None], gate_group, clipping=clipping)
            del gate, up
            down = quantize_weights(
                read_fp8_projection(source, index, layer, expert, "down")[None], down_group, clipping=clipping
            )
            filename = f"layer-{layer:02d}-expert-{expert:03d}.safetensors"
            save_expert(destination / filename, gate_up, down)
            tensor_bytes = sum(
                t.numel() * t.element_size() for b in (gate_up, down) for t in (b.codes, b.scales, b.sums, b.smoothing)
            )
            manifest["experts"][key] = {
                "file": filename,
                "sha256": file_sha256(destination / filename),
                "tensor_bytes": tensor_bytes,
                "gate_shape": list(gate_up.shape),
                "down_shape": list(down.shape),
            }
            with journal_path.open("a") as journal:
                journal.write(json.dumps({"key": key, "entry": manifest["experts"][key]}) + "\n")
            if len(manifest["experts"]) % 32 == 0:
                write_manifest(manifest_path, manifest)
            print(json.dumps({"layer": layer, "expert": expert, "bytes": tensor_bytes}), flush=True)
    manifest["complete"] = True
    write_manifest(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--experts", nargs="+", type=int, required=True)
    parser.add_argument("--gate-group", type=int, default=256)
    parser.add_argument("--down-group", type=int, default=256)
    parser.add_argument("--clipping", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    export(
        args.source,
        args.destination,
        args.layers,
        args.experts,
        args.gate_group,
        args.down_group,
        args.clipping,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
