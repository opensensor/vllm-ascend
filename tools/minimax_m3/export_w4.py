# SPDX-License-Identifier: Apache-2.0
"""Stream a local M3 checkpoint into resumable, independently checked W4 files.

Run as ``python -m tools.minimax_m3.export_w4 SOURCE DEST``. No NPU is used.
An incomplete export never receives a runnable manifest or marked config.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

from tools.minimax_m3.probe_w4 import quantize_chunk
from vllm_ascend._310p.ops.minimax_m3 import W4_BLOCK, W4_LAYOUT

MANIFEST = "minimax_m3_w4.json"
EXPERT_NAME = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
)
DTYPES = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4)}
MIXED_SOURCE_FORMAT = "minimax-m31-nvfp4-mxfp8-v1"


def mixed_weight_groups(weight_map):
    """Map every mixed-checkpoint tensor to exactly one logical output weight."""
    groups, consumed = {}, set()
    for name in sorted(weight_map):
        if name.endswith(".weight_packed"):
            base = name.removesuffix(".weight_packed")
            output = base + ".weight"
            if not EXPERT_NAME.fullmatch(output):
                raise ValueError("Only routed experts may use NVFP4")
            members = [
                base + suffix
                for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")
            ]
        elif name.endswith(".weight_scale_inv"):
            output = name.removesuffix("_scale_inv")
            if EXPERT_NAME.fullmatch(output):
                raise ValueError("M3.1 routed experts require NVFP4 companions")
            members = [output, name]
        else:
            continue
        if output in groups or any(n not in weight_map or n in consumed for n in members):
            raise ValueError("Missing or overlapping mixed quantization companions")
        groups[output] = members
        consumed.update(members)
    for name in weight_map.keys() - consumed:
        if name in groups or not name.endswith((".weight", ".bias", ".e_score_correction_bias")):
            raise ValueError(f"Unaccounted mixed-checkpoint tensor: {name}")
        groups[name] = [name]
    return groups


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_file(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"Expected a local file basename: {name!r}")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Symlinks are not accepted in a W4 export: {path}")
    return path


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def export_tensor(source, name, destination, *, rows=256, method="mse", mse_backend="torch", output_dtype=None):
    """Write one tensor using bounded row chunks, including large embeddings."""
    if rows <= 0 or rows % W4_BLOCK:
        raise ValueError("chunk rows must be a positive multiple of 32")
    shape = source.get_shape()
    source_dtype = source.get_dtype()
    if source_dtype not in DTYPES or not shape or any(d <= 0 for d in shape):
        raise ValueError(f"Unsupported tensor: {name}, {shape}, {source_dtype}")
    expert = EXPERT_NAME.fullmatch(name)
    if expert:
        if len(shape) != 2 or any(d % W4_BLOCK for d in shape):
            raise ValueError(f"Expert is not 32x32 tiled: {name}")
        descriptions = [
            ("codes", "U8", [shape[0], shape[1] // 2], 1),
            ("scales", "F16", [shape[0] // W4_BLOCK, shape[1] // W4_BLOCK], 2),
        ]
    else:
        output_dtype = output_dtype or ("F16" if source_dtype == "BF16" else source_dtype)
        if output_dtype not in DTYPES:
            raise ValueError("Unsupported output dtype")
        descriptions = [(name, output_dtype, shape, DTYPES[output_dtype][1])]
    header, offset = {}, 0
    for key, dtype, dims, size in descriptions:
        end = offset + math.prod(dims) * size
        header[key] = {"dtype": dtype, "shape": dims, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload_start = 8 + len(encoded)
    temporary = destination.with_suffix(".tmp")
    source_hash = hashlib.sha256()
    positions = {key: payload_start + meta["data_offsets"][0] for key, meta in header.items()}
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for start in range(0, shape[0], rows):
            weight = source[start : start + rows].contiguous()
            source_hash.update(weight.view(torch.uint8).numpy().tobytes())
            if not torch.isfinite(weight).all():
                raise ValueError(f"Nonfinite source tensor: {name}")
            if expert:
                codes, scales, reconstructed = quantize_chunk(weight, method, mse_backend=mse_backend)
                if not torch.isfinite(reconstructed.half()).all():
                    raise ValueError(f"FP16 expert reconstruction overflows: {name}")
                tensors = (("codes", codes), ("scales", scales))
            else:
                converted = weight.to(DTYPES[descriptions[0][1]][0])
                if not torch.isfinite(converted).all():
                    raise ValueError(f"FP16 conversion overflows: {name}")
                tensors = ((name, converted),)
            for key, tensor in tensors:
                data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                handle.seek(positions[key])
                handle.write(data)
                positions[key] += len(data)
        for key, meta in header.items():
            if positions[key] != payload_start + meta["data_offsets"][1]:
                raise ValueError(f"Incomplete tensor write: {name}")
    temporary.replace(destination)
    return {
        "file": destination.name,
        "sha256": file_digest(destination),
        "source_sha256": source_hash.hexdigest(),
        "source_shape": shape,
        "source_dtype": source_dtype,
        "tensors": header,
        "expert": bool(expert),
        "mse_backend": mse_backend if expert and method == "mse" else None,
    }


def export_checkpoint(source, destination, *, rows=256, method="mse", max_tensors=None, mse_backend="torch"):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Export must use a separate directory outside the source checkpoint")
    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    identity = {
        "layout": W4_LAYOUT,
        "method": method,
        "chunk_rows": rows,
        "source": str(source),
        "index_sha256": file_digest(index_path),
        "config_sha256": file_digest(config_path),
    }
    # The CPU backend preserves the canonical search/bytes and is deliberately
    # not part of resume identity. Record it per tensor for mixed-backend runs.
    identity["source_files"] = {
        name: {"size": local_file(source, name).stat().st_size, "mtime_ns": local_file(source, name).stat().st_mtime_ns}
        for name in sorted(set(weight_map.values()))
    }
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "export_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing export belongs to a different source or quantization recipe")
    elif any(destination.iterdir()):
        raise ValueError("Destination must be empty or contain a matching resumable export")
    else:
        atomic_json(identity_path, identity)
    records = {}
    for index, name in enumerate(sorted(weight_map)):
        if max_tensors is not None and index >= max_tensors:
            return {"complete": False, "tensor_count": len(records)}
        filename = f"tensor-{index:06d}.safetensors"
        target = destination / filename
        receipt = destination / f"tensor-{index:06d}.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["name"] != name or record["file"] != filename or file_digest(target) != record["sha256"]:
                raise ValueError(f"Corrupt export receipt or payload: {filename}")
        else:
            with safe_open(local_file(source, weight_map[name]), framework="pt", device="cpu") as handle:
                record = export_tensor(
                    handle.get_slice(name), name, target, rows=rows, method=method, mse_backend=mse_backend
                )
            record["name"] = name
            atomic_json(receipt, record)
        records[name] = record
        if index % 100 == 0:
            print(f"Exported/verified {index + 1}/{len(weight_map)} tensors", flush=True)
    # Assets are copied only after every weight is complete. Never copy a
    # source safetensors index that would reference the unconverted checkpoint.
    for path in source.iterdir():
        if path.is_file() and (
            path.suffix in (".json", ".model", ".txt", ".jinja", ".jinja2")
            or path.name.startswith(("LICENSE", "NOTICE", "README"))
        ):
            if path.name not in ("config.json", "model.safetensors.index.json"):
                shutil.copyfile(path, destination / path.name)
    config = json.loads(config_path.read_text())
    config["ascend_minimax_m3_w4"] = W4_LAYOUT
    if "text_config" in config:
        config["text_config"]["ascend_minimax_m3_w4"] = W4_LAYOUT
    for filename, expected in identity["source_files"].items():
        stat = local_file(source, filename).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Source checkpoint changed during conversion")
    shutil.copyfile(index_path, destination / "source_index.json")
    atomic_json(destination / "config.json", config)
    manifest = {
        **identity,
        "complete": True,
        "export_config_sha256": file_digest(destination / "config.json"),
        "tensor_count": len(records),
        "mse_backends": sorted({record.get("mse_backend", "torch") for record in records.values() if record["expert"]})
        if method == "mse"
        else [],
        "weights": records,
    }
    atomic_json(destination / MANIFEST, manifest)
    return {"complete": True, "tensor_count": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--method", choices=("minmax", "mse"), default="mse")
    parser.add_argument("--mse-backend", choices=("torch", "numba"), default="torch")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tensors", type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(
        json.dumps(
            export_checkpoint(
                args.source,
                args.destination,
                rows=args.chunk_rows,
                method=args.method,
                max_tensors=args.max_tensors,
                mse_backend=args.mse_backend,
            )
        )
    )


if __name__ == "__main__":
    main()
