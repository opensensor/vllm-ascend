# SPDX-License-Identifier: Apache-2.0
"""Budgeted, lossless W2/W3-to-W4 storage for permanent native checkpoints."""

import argparse
import json
import math
import os
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .fused_weight_layout import NZ_K, OUTPUT_TILE, _byte_fields, _byte_planes, _geometry, tensor_digest
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

MAX_HEADER_BYTES = 32 * 1024**2


def header(path):
    """Read metadata only, checking declared payload extents without mapping it."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= MAX_HEADER_BYTES or length > size - 8:
            raise ValueError("invalid safetensors header length")
        result = json.loads(handle.read(length))
    for name, value in result.items():
        if name != "__metadata__":
            start, stop = value["data_offsets"]
            if not 0 <= start <= stop <= size - length - 8:
                raise ValueError("safetensors payload is truncated")
    return result


def plan(source, banks=(), *, extra_bytes_per_rank, source_bits=3):
    """Whole-bank selection keeps every expert and both gate/up widths equal.

    The allowance is explicit extra *resident code* memory after reserving KV,
    graphs and workspace. It is not inferred from device utilization or disk
    space, and this function performs no device queries or code conversion.
    """
    if (
        type(extra_bytes_per_rank) is not int
        or extra_bytes_per_rank < 0
        or type(source_bits) is not int
        or source_bits not in (2, 3)
    ):
        raise ValueError("require a nonnegative integer byte allowance and source bits 2/3")
    source = Path(source).resolve()
    manifest = json.loads((source / MANIFEST).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True or manifest.get("layout") != LAYOUT:
        raise ValueError("W4 storage planning requires a complete permanent checkpoint")
    config, index, layers, experts = source_geometry(source)
    if (
        config.get("ascend_glm_expert_layout") != LAYOUT
        or manifest.get("layers") != layers
        or manifest.get("num_experts") != experts
    ):
        raise ValueError("permanent checkpoint manifest and geometry disagree")
    world = manifest["world_size"]
    if type(world) is not int or world <= 0 or experts % world:
        raise ValueError("invalid resident expert partition")
    text = config.get("text_config", config)
    hidden, inter = text["hidden_size"], text["moe_intermediate_size"]
    grouped, headers = defaultdict(list), {}
    native_files = {row["file"] for row in manifest["native_shards"]}
    for name, filename in index["weight_map"].items():
        match = CODE_NAME.fullmatch(name)
        if match is None:
            continue
        if filename not in native_files:
            raise ValueError("canonical code shard appears in the native index")
        if filename not in headers:
            headers[filename] = header(source / filename)
        metadata = headers[filename]
        if metadata.get("__metadata__", {}).get("layout") != LAYOUT:
            raise ValueError("code shard is missing its permanent layout marker")
        entry = metadata[name]
        n, k = (hidden, inter) if match[3] == "down" else (inter, hidden)
        shape = entry["shape"]
        if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("native code shape must have two positive integer dimensions")
        bits = shape[-1] * 8 // k
        if entry["dtype"] not in ("I8", "U8") or shape != [n, k * bits // 8] or bits not in (2, 3, 4):
            raise ValueError("native code tensor shape or dtype differs from configuration")
        old_bytes = math.prod(shape)
        if entry["data_offsets"][1] - entry["data_offsets"][0] != old_bytes:
            raise ValueError("native code payload size differs from shape")
        bank = f"layers.{match[1]}." + ("down" if match[3] == "down" else "gate_up")
        grouped[bank].append(dict(name=name, bits=bits, bytes=old_bytes, expanded_bytes=n * k // 2, k=k))
    selected = set(banks)
    if selected - grouped.keys():
        raise ValueError("unknown whole bank: " + ", ".join(sorted(selected - grouped.keys())))
    records, added = [], [0] * world
    for bank, tensors in sorted(grouped.items()):
        widths = {row["bits"] for row in tensors}
        required = experts * (1 if bank.endswith(".down") else 2)
        if len(tensors) != required or len(widths) != 1:
            raise ValueError("resident bank has mixed widths or incomplete expert coverage")
        bits = widths.pop()
        enabled = bank in selected
        if enabled and bits != source_bits:
            raise ValueError("selected bank does not use the requested source width")
        delta = sum(row["expanded_bytes"] - row["bytes"] for row in tensors)
        per_rank = [0] * world
        for row in tensors:
            rank = int(CODE_NAME.fullmatch(row["name"])[2]) // (experts // world)
            per_rank[rank] += row["expanded_bytes"] - row["bytes"]
        if enabled:
            added = [left + right for left, right in zip(added, per_rank)]
        records.append(
            dict(
                bank=bank,
                source_bits=bits,
                eligible=bits == source_bits,
                selected=enabled,
                tensor_count=required,
                source_code_bytes=sum(row["bytes"] for row in tensors),
                extra_code_bytes=delta,
                extra_bytes_per_rank=per_rank,
                new_payload_bytes=sum(row["expanded_bytes"] for row in tensors),
                tensors=tensors if enabled else [],
            )
        )
    return dict(
        kind="offline_lossless_w4_storage_plan",
        source=str(source),
        source_index_sha256=file_digest(source / INDEX),
        world_size=world,
        num_experts=experts,
        layers=layers,
        source_bits=source_bits,
        banks=records,
        additional_bytes_per_rank=added,
        allowance_bytes_per_rank=extra_bytes_per_rank,
        fits_allowance=all(value <= extra_bytes_per_rank for value in added),
        new_payload_bytes=sum(row["new_payload_bytes"] for row in records if row["selected"]),
        disk_usage_note="unchanged source shards remain hard-linked; new payload excludes headers",
        precision_changed=False,
        cube_operation_count_changed=False,
        hardware_validation="not_run",
    )


def promote(packed, k):
    """Widen physical signed fields to nibbles without float quantization."""
    e, n, bits = _geometry(packed, k)
    if packed.device.type != "cpu" or bits not in (2, 3):
        raise ValueError("promotion requires CPU native W2/W3 byte banks")
    raw = packed.numpy().view(np.uint8)
    values = _byte_fields(raw.reshape(e, n // OUTPUT_TILE, k // NZ_K, 3 if bits == 3 else 1, -1), bits)
    values = values.reshape(e, n // OUTPUT_TILE, k // NZ_K, -1)
    sign = 1 << (bits - 1)
    nibbles = np.where(values >= sign, values + (16 - (1 << bits)), values).astype(np.uint8)
    pairs = nibbles.reshape(e, n // OUTPUT_TILE, k // NZ_K, -1, 2)
    widened = np.ascontiguousarray(pairs[..., 0] | (pairs[..., 1] << 4)).reshape(e, n, k // 2)
    # Independent inverse packing plus a signed-range check distinguishes true
    # sign extension from merely copying the old low bits into a W4 field.
    decoded = np.stack((widened & 15, widened >> 4), -1).reshape(e, n // OUTPUT_TILE, k // NZ_K, -1)
    if not np.all((decoded < sign) | (decoded >= 16 - sign)):
        raise ValueError("promoted nibble exceeds the original signed code range")
    inverse = _byte_planes(decoded & ((1 << bits) - 1), bits).reshape(raw.shape)
    if not np.array_equal(inverse, raw):
        raise ValueError("promotion changed original packed code values")
    return torch.from_numpy(widened.view(np.int8 if packed.dtype == torch.int8 else np.uint8))


def export(source, output, banks, *, extra_bytes_per_rank, source_bits=3):
    """Append a budgeted checkpoint variant; keep scales and kernels unchanged."""
    source, output = Path(source).resolve(), Path(output).resolve()
    banks = sorted(set(banks))
    if output.exists():
        raise FileExistsError(output)
    proposal = plan(source, banks, extra_bytes_per_rank=extra_bytes_per_rank, source_bits=source_bits)
    if not banks or not proposal["fits_allowance"]:
        raise ValueError("select at least one bank and satisfy the extra resident-memory allowance")
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("unchanged shards require the source filesystem")
    parent = json.loads((source / MANIFEST).read_text())
    index = json.loads((source / INDEX).read_text())
    bundles = {}
    for key in ("kernel_bundle", "indexer_kernel_bundle"):
        if key in parent:
            relative = Path(parent[key])
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("checkpoint bundle must be a relative child path")
            if not (source / relative).is_dir():
                raise FileNotFoundError(source / relative)
            bundles[key] = relative
    provenance, _ = kernel_assets(source / bundles["kernel_bundle"])
    if file_digest(source / bundles["kernel_bundle"] / "provenance.json") != parent["kernel_provenance_sha256"]:
        raise ValueError("checkpoint kernel provenance checksum differs")
    config = json.loads((source / "config.json").read_text())
    validate_scale_contract(parent, config.get("ascend_glm_expert_scale_layout"), provenance["_build"])
    partitions = defaultdict(list)
    for bank in proposal["banks"]:
        for row in bank["tensors"]:
            match = CODE_NAME.fullmatch(row["name"])
            rank = int(match[2]) // (proposal["num_experts"] // proposal["world_size"])
            partitions[int(match[1]), rank].append(row)
    filenames = {(layer, rank): f"w4-storage-rank{rank}-layer{layer:03d}.safetensors" for layer, rank in partitions}
    if any(filename in index["weight_map"].values() for filename in filenames.values()):
        raise ValueError("storage promotion shard already exists in the parent index")
    # Publish failure state before copying; only the final write admits loading.
    output.mkdir()
    manifest = dict(
        parent,
        complete=False,
        source=str(source),
        source_index_sha256=proposal["source_index_sha256"],
        parent_native_manifest_sha256=file_digest(source / MANIFEST),
        storage_promotion=proposal,
        hardware_validation="not_run",
        quality_validation="not_run",
    )
    write_json(output / MANIFEST, manifest)
    for filename in sorted(set(index["weight_map"].values())):
        os.link(source / filename, output / filename)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (INDEX, MANIFEST):
            shutil.copy2(path, output / path.name)
    for key, relative in bundles.items():
        if key == "kernel_bundle":
            copy_kernel_bundle(source / relative, output / relative)
        else:
            shutil.copytree(source / relative, output / relative)
    records = []
    for (layer, rank), rows in sorted(partitions.items()):
        tensors, hashes = {}, {}
        by_shard = defaultdict(list)
        for row in rows:
            by_shard[index["weight_map"][row["name"]]].append(row)
        for shard, entries in sorted(by_shard.items()):
            with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
                for row in entries:
                    value = handle.get_tensor(row["name"])
                    hashes[row["name"]] = tensor_digest(value)
                    tensors[row["name"]] = promote(value[None], row["k"])[0]
        filename = filenames[layer, rank]
        path = output / filename
        temporary = path.with_suffix(".tmp")
        save_file(tensors, str(temporary), metadata={"layout": LAYOUT, "storage_promotion": "lossless_w4_v1"})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name, expected in tensors.items():
                if not torch.equal(handle.get_tensor(name), expected):
                    raise ValueError("promoted shard readback changed bytes")
                index["weight_map"][name] = filename
        records.append(
            dict(
                file=filename,
                layer=layer,
                rank=rank,
                sha256=file_digest(path),
                bytes=path.stat().st_size,
                tensor_count=len(tensors),
                source_tensor_sha256=hashes,
            )
        )
    if sum(row["tensor_count"] for row in records) != sum(len(rows) for rows in partitions.values()):
        raise ValueError("storage export did not cover every selected expert tensor")
    previous_banks = index["metadata"].get("lossless_w4_storage_banks", [])
    index["metadata"]["lossless_w4_storage_banks"] = sorted(set(previous_banks) | set(banks))
    write_json(output / INDEX, index)
    manifest.update(
        complete=True,
        native_shards=[*parent["native_shards"], *records],
        storage_promotion_shards=[*parent.get("storage_promotion_shards", []), *records],
    )
    write_json(output / MANIFEST, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "export"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bank", action="append", default=[])
    parser.add_argument("--extra-bytes-per-rank", type=int, required=True)
    parser.add_argument("--source-bits", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if args.action == "export":
        if args.output is None:
            parser.error("export requires --output")
        result = export(
            args.source,
            args.output,
            args.bank,
            extra_bytes_per_rank=args.extra_bytes_per_rank,
            source_bits=args.source_bits,
        )
    else:
        result = plan(
            args.source, args.bank, extra_bytes_per_rank=args.extra_bytes_per_rank, source_bits=args.source_bits
        )
    if args.action == "plan" and args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
