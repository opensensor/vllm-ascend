"""Rebuild one failed shard from checksum-verified source; never repair checksums alone."""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.glm_perf.fused_weight_layout import tensor_digest
from tools.glm_perf.native_checkpoint import file_digest, write_json
from tools.glm_perf.w4_storage_checkpoint import promote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("filename")
    args = parser.parse_args()
    assert args.filename.startswith("w4-storage-") and Path(args.filename).name == args.filename
    original_name = args.filename.removeprefix("w4-storage-")
    source_manifest = json.loads((args.source / "native-layout.json").read_text())
    manifest_path = args.output / "native-layout.json"
    manifest = json.loads(manifest_path.read_text())
    source_record = next(row for row in source_manifest["native_shards"] if row["file"] == original_name)
    record = next(row for row in manifest["storage_promotion_shards"] if row["file"] == args.filename)
    assert file_digest(args.source / original_name) == source_record["sha256"]
    selected = {
        row["name"]: row
        for bank in manifest["storage_promotion"]["banks"]
        for row in bank["tensors"]
        if row["name"] in record["source_tensor_sha256"]
    }
    assert selected
    manifest["complete"] = False
    write_json(manifest_path, manifest)
    tensors = {}
    with safe_open(str(args.source / original_name), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {}, storage_promotion="lossless_w4_v1")
        names = handle.keys()
        for name in names:
            value = handle.get_tensor(name)
            if name in selected:
                assert tensor_digest(value) == record["source_tensor_sha256"][name]
                value = promote(value[None], selected[name]["k"])[0]
            tensors[name] = value
    target = args.output / args.filename
    temporary = target.with_suffix(".repair.tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(target)
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        assert handle.metadata() == metadata and set(handle.keys()) == tensors.keys()
        for name, value in tensors.items():
            assert torch.equal(value, handle.get_tensor(name)), name
    digest = file_digest(target)
    for field in ("native_shards", "scale_shards", "storage_promotion_shards"):
        for row in manifest.get(field, []):
            if row["file"] == args.filename:
                row.update(sha256=digest, bytes=target.stat().st_size)
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "rebuilt": args.filename,
                "source_sha256": source_record["sha256"],
                "tensor_count": len(tensors),
                "sha256": digest,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
