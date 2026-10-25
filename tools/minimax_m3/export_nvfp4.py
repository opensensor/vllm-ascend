# SPDX-License-Identifier: Apache-2.0
"""Resumable source-preserving expert-bank export for a controlled model trial."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.minimax_m3.export_w4 import atomic_json, file_digest, local_file
from tools.minimax_m3.nvfp4 import LOOKUP_SIZE, NVFP4_LAYOUT, pack_source

MANIFEST = "nvfp4_experts.json"
PROJECTIONS = ("w1", "w3", "w2")


def export(source, destination, ep_size=2):
    source, destination = source.resolve(strict=True), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Use a separate destination outside the original checkpoint")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("moe_quant_format") != "nvfp4-pack-quantized" or quant.get("group_size") != 16:
        raise ValueError("Unsupported source NVFP4 quantization contract")
    text = config["text_config"]
    if ep_size <= 0 or text["num_local_experts"] % ep_size:
        raise ValueError("Expert count must divide EP size")
    local_experts = text["num_local_experts"] // ep_size
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    identity = {
        "layout": NVFP4_LAYOUT,
        "ep_size": ep_size,
        "config_sha256": file_digest(source / "config.json"),
        "index_sha256": file_digest(index_path),
        "source_files": {
            name: {
                "size": local_file(source, name).stat().st_size,
                "mtime_ns": local_file(source, name).stat().st_mtime_ns,
            }
            for name in sorted(set(index.values()))
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Destination contains a different source or layout")
    elif any(destination.iterdir()):
        raise ValueError("Destination is not an empty or resumable NVFP4 export")
    else:
        atomic_json(identity_path, identity)
    records = {}
    with ExitStack() as stack:
        handles = {}

        def tensor(name):
            shard = index[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(local_file(source, shard), framework="pt", device="cpu"))
            return handles[shard].get_tensor(name)

        for layer, enabled in enumerate(text["moe_layer_freq"]):
            if not enabled:
                continue
            for rank in range(ep_size):
                for projection in PROJECTIONS:
                    key = f"layer-{layer:02d}-ep-{rank}-{projection}"
                    target = destination / (key + ".safetensors")
                    receipt = destination / (key + ".json")
                    if receipt.exists():
                        record = json.loads(receipt.read_text())
                        if record["file"] != target.name or record["sha256"] != file_digest(target):
                            raise ValueError("Corrupt source-preserving expert bank")
                    else:
                        rows, cols = text["intermediate_size"], text["hidden_size"]
                        if projection == "w2":
                            rows, cols = cols, rows
                        data = {
                            "codes": torch.empty(local_experts, rows, cols // 2, dtype=torch.int8),
                            "scales": torch.empty(local_experts, rows, cols // 16, dtype=torch.uint8),
                            "lookup": torch.empty(local_experts, LOOKUP_SIZE, dtype=torch.float16),
                            "weight_global_scale": torch.empty(local_experts),
                            "input_global_scale": torch.empty(local_experts),
                        }
                        for local in range(local_experts):
                            expert = rank * local_experts + local
                            base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}"
                            weight_scale, input_scale = (
                                tensor(base + ".weight_global_scale"),
                                tensor(base + ".input_global_scale"),
                            )
                            if not torch.isfinite(input_scale).all() or not (input_scale > 0).all():
                                raise ValueError("Invalid source activation scale")
                            bank = pack_source(
                                tensor(base + ".weight_packed"), tensor(base + ".weight_scale"), weight_scale
                            )
                            for name, value in zip(("codes", "scales", "lookup"), bank):
                                data[name][local].copy_(value)
                            data["weight_global_scale"][local] = weight_scale.reshape(())
                            data["input_global_scale"][local] = input_scale.reshape(())
                        temporary = target.with_suffix(".tmp")
                        save_file(data, temporary)
                        temporary.replace(target)
                        record = {
                            "file": target.name,
                            "sha256": file_digest(target),
                            "layer": layer,
                            "ep_rank": rank,
                            "projection": projection,
                            "experts": local_experts,
                            "shape": [rows, cols],
                            "bytes": target.stat().st_size,
                        }
                        atomic_json(receipt, record)
                        del data
                    records[key] = record
                    print(
                        json.dumps({"bank": key, "complete_banks": len(records), "bytes": record["bytes"]}), flush=True
                    )
    for name, expected in identity["source_files"].items():
        stat = (source / name).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError("Source changed during export")
    manifest = {
        **identity,
        "complete": True,
        "banks": records,
        "payload_bytes": sum(r["bytes"] for r in records.values()),
    }
    atomic_json(destination / MANIFEST, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        result = export(args.source, args.destination, args.ep_size)
    print(json.dumps({"complete": True, "banks": len(result["banks"]), "bytes": result["payload_bytes"]}))


if __name__ == "__main__":
    main()
