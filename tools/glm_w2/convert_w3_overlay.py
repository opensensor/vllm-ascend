# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert selected GLM routed layers to packed W3 over an existing checkpoint.

The output checkpoint reuses the base shards through symlinks and indexes only
the W3 replacements from each overlay shard. Use ``--relink-base`` after moving
the candidate to a host where the base checkpoint has a different path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from shutil import copy2

from tools.deepseek_w2.convert_full import (
    DEFAULT_CHUNK_ROWS,
    INDEX_FILE,
    SafetensorsShardReader,
    _drop_page_cache,
    _load_headers,
    _save_shard_atomic,
    _sha256_file,
    _write_json_atomic,
)
from tools.glm_w2.convert_full import convert_item, plan_items

DEFAULT_W3_LAYERS = frozenset({8, 9, 10, 12, 13, 14, 16, 17})
MANIFEST_FILE = "w3_overlay_manifest.json"


def _weight_map_sha256(weight_map: dict[str, str]) -> str:
    canonical = json.dumps(weight_map, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _selected_bytes(root: Path, weight_map: dict[str, str]) -> int:
    headers = _load_headers(root, weight_map)
    return sum(
        headers[weight_map[name]][name]["data_offsets"][1] - headers[weight_map[name]][name]["data_offsets"][0]
        for name in weight_map
    )


def relink_base(out_dir: Path, base_dir: Path, base_files: set[str]) -> None:
    """Point base shard links at a relocated checkpoint."""
    for name in base_files:
        target = (base_dir / name).resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        link = out_dir / name
        if link.exists() and not link.is_symlink():
            raise FileExistsError(f"refusing to replace regular file {link}")
        link.unlink(missing_ok=True)
        link.symlink_to(target)


def convert_overlay(
    source_dir: Path,
    base_dir: Path,
    out_dir: Path,
    layers: frozenset[int],
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict:
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("W3 layer selection must contain nonnegative indices")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    source_map = json.loads((source_dir / INDEX_FILE).read_text())["weight_map"]
    base_map = json.loads((base_dir / INDEX_FILE).read_text())["weight_map"]
    selected = {
        name: shard
        for name, shard in source_map.items()
        if any(f".layers.{layer}.mlp.experts." in name for layer in layers)
    }
    headers = _load_headers(source_dir, selected)
    items, excluded = plan_items(selected, headers, 1 << 40, w3_layers=layers)
    if excluded or any(item.target != "W3" or item.n_parts != 1 for item in items):
        raise ValueError("selected source tensors must be unsplit W3 routed experts")
    by_layer = defaultdict(list)
    for item in items:
        layer = int(item.weight_name.split(".layers.", 1)[1].split(".", 1)[0])
        by_layer[layer].append(item)
    if set(by_layer) != set(layers):
        raise ValueError("one or more requested layers have no source expert weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_map = dict(base_map)
    overlay_files: dict[str, dict] = {}
    for layer in sorted(layers):
        shard_name = f"w3-overlay-layer-{layer:02d}.safetensors"
        layer_items = by_layer[layer]
        expected_names = {output.name for item in layer_items for output in item.outputs}
        base_layer_names = {name for name in base_map if f".layers.{layer}.mlp.experts." in name}
        if expected_names != base_layer_names:
            missing = base_layer_names - expected_names
            extra = expected_names - base_layer_names
            raise ValueError(
                f"layer {layer} source/base expert coverage differs: {len(missing)} missing, {len(extra)} extra"
            )
        for name in expected_names:
            out_map[name] = shard_name
        shard_path = out_dir / shard_name
        if shard_path.exists():
            saved = _load_headers(out_dir, {name: shard_name for name in expected_names})[shard_name]
            if set(saved) != expected_names:
                raise ValueError(f"incomplete existing W3 overlay shard {shard_path}")
            for item in layer_items:
                for output in item.outputs:
                    descriptor = saved[output.name]
                    offset_start, offset_end = descriptor["data_offsets"]
                    if descriptor["dtype"] != output.dtype or offset_end - offset_start != output.nbytes:
                        raise ValueError(f"invalid existing W3 overlay tensor {output.name}")
            sha = _sha256_file(shard_path)
            nbytes = shard_path.stat().st_size
        else:
            readers: dict[str, SafetensorsShardReader] = {}
            tensors = {}
            for item in layer_items:
                source_shard = selected[item.weight_name]
                reader = readers.get(source_shard)
                if reader is None:
                    reader = readers[source_shard] = SafetensorsShardReader(source_dir / source_shard)
                output, _ = convert_item(reader, item, chunk_rows)
                tensors.update(output)
            sha, nbytes = _save_shard_atomic(out_dir, shard_name, tensors)
            del tensors, readers
            for source_shard in {selected[item.weight_name] for item in layer_items}:
                _drop_page_cache(source_dir / source_shard)
            _drop_page_cache(shard_path)
        overlay_files[shard_name] = {
            "layer": layer,
            "sha256": sha,
            "file_bytes": nbytes,
            "tensors": len(expected_names),
        }
        print(f"W3 layer {layer}: {shard_name}, {nbytes} bytes", flush=True)

    base_files = set(base_map.values())
    if base_files & overlay_files.keys():
        raise ValueError("W3 overlay shard name collides with a base shard")
    relink_base(out_dir, base_dir, base_files)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        source = base_dir / name
        if source.exists():
            destination = out_dir / name
            if destination.is_symlink():
                destination.unlink()
            copy2(source, destination)
    total_size = _selected_bytes(out_dir, out_map)
    _write_json_atomic(
        out_dir / INDEX_FILE,
        {
            "metadata": {"total_size": total_size, "converter": "tools/glm_w2/convert_w3_overlay.py"},
            "weight_map": dict(sorted(out_map.items())),
        },
    )
    manifest = {
        "schema_version": 1,
        "source_dir": str(source_dir),
        "source_index_sha256": _sha256_file(source_dir / INDEX_FILE),
        "base_dir": str(base_dir),
        "base_index_sha256": _sha256_file(base_dir / INDEX_FILE),
        "base_weight_map_sha256": _weight_map_sha256(base_map),
        "layers": sorted(layers),
        "format": "signed W3, eight codes in three little-endian bytes, FP32 [32,32] no-clip scales",
        "selected_bytes": total_size,
        "base_files": sorted(base_files),
        "overlay_files": overlay_files,
    }
    _write_json_atomic(out_dir / MANIFEST_FILE, manifest)
    return manifest


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default=",".join(map(str, sorted(DEFAULT_W3_LAYERS))))
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--relink-base", action="store_true")
    args = parser.parse_args()
    if args.relink_base:
        base_map = json.loads((args.base_dir / INDEX_FILE).read_text())["weight_map"]
        manifest_path = args.out_dir / MANIFEST_FILE
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected_map_hash = manifest.get("base_weight_map_sha256")
            if expected_map_hash and _weight_map_sha256(base_map) != expected_map_hash:
                raise ValueError("relocated base checkpoint has a different safetensors weight map")
        relink_base(args.out_dir, args.base_dir, set(base_map.values()))
        if manifest_path.exists():
            manifest["base_dir"] = str(args.base_dir)
            _write_json_atomic(manifest_path, manifest)
        return 0
    if args.source_dir is None:
        parser.error("--source-dir is required unless --relink-base is set")
    layers = frozenset(int(part) for part in args.layers.split(",") if part.strip())
    result = convert_overlay(args.source_dir, args.base_dir, args.out_dir, layers, args.chunk_rows)
    print(
        json.dumps(
            {
                "layers": result["layers"],
                "selected_bytes": result["selected_bytes"],
                "overlay_files": len(result["overlay_files"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
