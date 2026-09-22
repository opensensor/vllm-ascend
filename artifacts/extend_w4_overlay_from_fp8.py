#!/usr/bin/env python3
"""Extend an existing mixed-W4 overlay with selected FP8 expert layers."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors.torch import save_file

from tools.deepseek_w2.w2_convert import SafetensorsShardReader
from tools.glm_w2.convert_full import (
    _expert_layer_index,
    _load_headers,
    convert_item,
    plan_items,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_overlay", type=Path)
    parser.add_argument("fp8_source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    args = parser.parse_args()

    layers = frozenset(int(value) for value in args.layers.split(","))
    base_index = json.loads((args.base_overlay / "model.safetensors.index.json").read_text())
    source_index = json.loads((args.fp8_source / "model.safetensors.index.json").read_text())
    source_map = source_index["weight_map"]
    headers = _load_headers(args.fp8_source, source_map)
    items, _ = plan_items(
        source_map,
        headers,
        5 * (1 << 30),
        default_expert_bits=2,
        w4_layers=layers,
    )
    items_by_layer = defaultdict(list)
    for item in items:
        layer = _expert_layer_index(item.weight_name)
        if item.family == "routed_expert_weight" and layer in layers:
            if item.target != "W4" or item.n_parts != 1:
                raise RuntimeError(f"unexpected conversion plan for {item.weight_name}")
            items_by_layer[layer].append(item)

    if set(items_by_layer) != set(layers):
        raise RuntimeError(f"planned layers {sorted(items_by_layer)} != requested {sorted(layers)}")

    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.base_overlay.iterdir():
        if path.name == "model.safetensors.index.json":
            continue
        destination = args.output / path.name
        if path.name.startswith("model-") or path.name.startswith("w4-overlay-"):
            if not destination.exists():
                destination.symlink_to(path.resolve())
        elif path.is_file():
            shutil.copy2(path, destination)

    output_map = dict(base_index["weight_map"])
    new_overlay_bytes = 0
    for layer in sorted(layers):
        output_name = f"w4-overlay-layer-{layer:02d}.safetensors"
        output_path = args.output / output_name
        if output_path.exists():
            raise RuntimeError(f"refusing to overwrite {output_path}")

        readers: dict[str, SafetensorsShardReader] = {}
        tensors = {}
        try:
            for item in items_by_layer[layer]:
                shard = source_map[item.weight_name]
                reader = readers.setdefault(
                    shard,
                    SafetensorsShardReader(args.fp8_source / shard),
                )
                converted, _ = convert_item(reader, item, args.chunk_rows)
                tensors.update(converted)
        finally:
            readers.clear()

        save_file(tensors, output_path)
        new_overlay_bytes += output_path.stat().st_size
        for name in tensors:
            if name not in output_map:
                raise RuntimeError(f"converted tensor is absent from base overlay: {name}")
            output_map[name] = output_name
        print(f"layer {layer}: {len(tensors)} tensors -> {output_name}", flush=True)

    output_index = dict(base_index)
    output_index["weight_map"] = output_map
    output_index.setdefault("metadata", {})["extended_w4_layers"] = sorted(layers)
    output_index["metadata"]["new_overlay_bytes"] = new_overlay_bytes
    (args.output / "model.safetensors.index.json").write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
