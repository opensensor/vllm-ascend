# SPDX-License-Identifier: Apache-2.0
"""Resumable M3.1 NVFP4/MXFP8 to signed block32 W4 / FP16 conversion.

Reads local shards in bounded row chunks. The original quantization metadata
is retained as provenance, never advertised as the converted runtime format.
"""

import argparse
import copy
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from tools.minimax_m3.export_w4 import (
    EXPERT_NAME,
    MANIFEST,
    MIXED_SOURCE_FORMAT,
    atomic_json,
    export_tensor,
    file_digest,
    local_file,
    mixed_weight_groups,
)
from tools.minimax_m3.mixed_checkpoint import dequant_mxfp8, dequant_nvfp4
from vllm_ascend._310p.ops.minimax_m3 import W4_LAYOUT


class DecodedRows:
    """Expose a logical FP32 matrix to the existing bounded W4 exporter."""

    def __init__(self, slices, tensors, packed):
        self.slices, self.tensors, self.packed = slices, tensors, packed

    def get_shape(self):
        shape = list(self.slices[0].get_shape())
        if len(shape) != 2:
            raise ValueError("Mixed quantization requires matrices")
        if self.packed:
            shape[1] *= 2
        return shape

    def get_dtype(self):
        return "F32"

    def __getitem__(self, rows):
        weight, scales = (s[rows] for s in self.slices)
        if self.packed:
            return dequant_nvfp4(weight, scales, self.tensors[0])
        return dequant_mxfp8(weight, scales)


def export_checkpoint(source, destination, *, rows=256, mse_backend="numba", max_tensors=None):
    if rows <= 0 or rows % 32 or (max_tensors is not None and max_tensors < 0):
        raise ValueError("Invalid row chunk or tensor limit")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Export requires a separate destination outside the source")
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    quant = config.get("quantization_config", {})
    if (
        quant.get("quant_method") != "mxfp8"
        or quant.get("moe_quant_format") != "nvfp4-pack-quantized"
        or quant.get("group_size") != 16
        or quant.get("weight_block_size") != [1, 32]
    ):
        raise ValueError("Unsupported mixed source quantization contract")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    groups = mixed_weight_groups(weight_map)
    identity = {
        "source_format": MIXED_SOURCE_FORMAT,
        "layout": W4_LAYOUT,
        "method": "mse",
        "chunk_rows": rows,
        "source": str(source),
        "index_sha256": file_digest(index_path),
        "config_sha256": file_digest(config_path),
        "source_files": {
            name: {
                "size": local_file(source, name).stat().st_size,
                "mtime_ns": local_file(source, name).stat().st_mtime_ns,
            }
            for name in sorted(set(weight_map.values()))
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "export_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing export belongs to a different source or recipe")
    elif any(destination.iterdir()):
        raise ValueError("Destination must be empty or a matching resumable export")
    else:
        atomic_json(identity_path, identity)
    records = {}
    for index, name in enumerate(sorted(groups)):
        if max_tensors is not None and index >= max_tensors:
            return {"complete": False, "tensor_count": len(records)}
        members = groups[name]
        target = destination / f"tensor-{index:06d}.safetensors"
        receipt = destination / f"tensor-{index:06d}.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if (
                record["name"] != name
                or record["file"] != target.name
                or record["source_tensors"] != members
                or file_digest(target) != record["sha256"]
            ):
                raise ValueError("Corrupt mixed export receipt or payload")
        else:
            with ExitStack() as stack:
                handles = {
                    shard: stack.enter_context(safe_open(local_file(source, shard), framework="pt", device="cpu"))
                    for shard in {weight_map[n] for n in members}
                }
                views = [handles[weight_map[n]].get_slice(n) for n in members]
                output_dtype = None
                if len(members) == 1:
                    decoded = views[0]
                elif len(members) == 2:
                    decoded = DecodedRows(views, [], packed=False)
                    output_dtype = "F16"
                else:
                    globals_ = [handles[weight_map[n]].get_tensor(n) for n in members[2:]]
                    if any(
                        v.dtype != torch.float32 or v.numel() != 1 or not torch.isfinite(v).all() or not (v > 0).all()
                        for v in globals_
                    ):
                        raise ValueError("Invalid NVFP4 weight/input global scale")
                    decoded = DecodedRows(views[:2], globals_, packed=True)
                    text = config["text_config"]
                    shape = [text["intermediate_size"], text["hidden_size"]]
                    if EXPERT_NAME.fullmatch(name)[3] == "w2":
                        shape.reverse()
                    if decoded.get_shape() != shape:
                        raise ValueError("Expert geometry differs from config")
                record = export_tensor(
                    decoded, name, target, rows=rows, method="mse", mse_backend=mse_backend, output_dtype=output_dtype
                )
            record.update(name=name, source_tensors=members)
            atomic_json(receipt, record)
        records[name] = record
        if index % 100 == 0:
            print(f"Exported/verified {index + 1}/{len(groups)} tensors", flush=True)
    for filename, expected in identity["source_files"].items():
        stat = local_file(source, filename).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError("Source changed during conversion")
    for path in source.iterdir():
        if path.is_file() and (
            path.suffix in (".json", ".model", ".txt", ".jinja", ".jinja2")
            or path.name.startswith(("LICENSE", "NOTICE", "README"))
        ):
            if path.name not in ("config.json", "model.safetensors.index.json"):
                shutil.copyfile(path, destination / path.name)
    converted = copy.deepcopy(config)
    converted.pop("quantization_config", None)
    converted["text_config"].pop("quantization_config", None)
    converted["torch_dtype"] = "float16"
    converted["ascend_minimax_m3_w4"] = W4_LAYOUT
    converted["text_config"]["ascend_minimax_m3_w4"] = W4_LAYOUT
    shutil.copyfile(index_path, destination / "source_index.json")
    shutil.copyfile(config_path, destination / "source_config.json")
    atomic_json(destination / "config.json", converted)
    manifest = {
        **identity,
        "complete": True,
        "export_config_sha256": file_digest(destination / "config.json"),
        "tensor_count": len(records),
        "weights": records,
        "source_hash_semantics": "Per-record source_sha256 hashes decoded FP32 rows for quantized matrices",
        "mse_backends": sorted({r["mse_backend"] for r in records.values() if r["expert"]}),
    }
    # Require complete configured expert coverage before publishing a manifest.
    expected = {
        (layer, expert, projection)
        for layer, enabled in enumerate(config["text_config"]["moe_layer_freq"])
        if enabled
        for expert in range(config["text_config"]["num_local_experts"])
        for projection in ("w1", "w2", "w3")
    }
    observed = {(int(m[1]), int(m[2]), m[3]) for name in records if (m := EXPERT_NAME.fullmatch(name))}
    if observed != expected:
        raise ValueError("Incomplete configured expert coverage")
    atomic_json(destination / MANIFEST, manifest)
    return {"complete": True, "tensor_count": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mse-backend", choices=("numba", "torch"), default="numba")
    parser.add_argument("--max-tensors", type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(
        json.dumps(
            export_checkpoint(
                args.source,
                args.destination,
                rows=args.chunk_rows,
                mse_backend=args.mse_backend,
                max_tensors=args.max_tensors,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
