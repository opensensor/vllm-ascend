# SPDX-License-Identifier: Apache-2.0
"""Verify every permanent expert's SHA and native header without loading weights."""

import argparse
import json
import time
from pathlib import Path

from safetensors import safe_open

from .channel_scale_checkpoint import file_sha256
from .channel_scale_layout import FACTOR_LAYOUT
from .channel_scale_math import LAYOUT


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("complete") is not True
        or manifest.get("layout") != LAYOUT
        or manifest.get("factor_layout") != FACTOR_LAYOUT
    ):
        raise ValueError("requires a complete native channel-scale overlay")
    selected = {f"{layer}:{expert}" for layer in manifest["layers"] for expert in manifest["expert_ids"]}
    if set(manifest["experts"]) != selected or not selected:
        raise ValueError("permanent overlay coverage differs from its selections")
    files, tensor_bytes, file_bytes = set(), 0, 0
    started = time.monotonic()
    for key in sorted(selected):
        entry = manifest["experts"][key]
        filename = entry["file"]
        if Path(filename).name != filename or filename in files:
            raise ValueError("unsafe or aliased permanent expert shard")
        files.add(filename)
        path = root / filename
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("permanent expert checksum differs: " + key)
        expected = {}
        for prefix, shape, group in (
            ("gate_up", entry["gate_shape"], manifest["gate_group"]),
            ("down", entry["down_shape"], manifest["down_group"]),
        ):
            e, n, k = shape
            if e != 1 or n % 128 or k % group or k % 256:
                raise ValueError("unsupported permanent expert header geometry")
            expected[prefix + ".codes"] = ([e, n, k // 2], "I8", 1)
            expected[prefix + ".scales"] = ([e, n // 128, k // group, 128], "F32", 4)
            expected[prefix + ".sums"] = ([e, n // 128, k // group, 128], "I32", 4)
            expected[prefix + ".smoothing"] = ([e, k], "F32", 4)
        shard_bytes = 0
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("layout") != LAYOUT
                or metadata.get("factor_layout") != FACTOR_LAYOUT
                or metadata.get("source") != "original_fp8_block128"
                or metadata.get("gate_group") != str(manifest["gate_group"])
                or metadata.get("down_group") != str(manifest["down_group"])
                or set(handle.keys()) != set(expected)
            ):
                raise ValueError("permanent native header contract differs: " + key)
            for name, (shape, dtype, size) in expected.items():
                value = handle.get_slice(name)
                if value.get_shape() != shape or value.get_dtype() != dtype:
                    raise ValueError("permanent tensor header differs: " + name)
                count = 1
                for dimension in shape:
                    count *= dimension
                shard_bytes += count * size
        if shard_bytes != entry["tensor_bytes"]:
            raise ValueError("permanent tensor byte count differs: " + key)
        tensor_bytes += shard_bytes
        file_bytes += path.stat().st_size
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError("overlay manifest changed during verification")
    return {
        "status": "all selected shard SHA and native headers verified on CPU",
        "qualification": "storage integrity only; no NPU or full-model quality gate",
        "overlay": str(root),
        "manifest_sha256": manifest_hash,
        "source_index_sha256": manifest["source_index_sha256"],
        "shards": len(files),
        "layers": manifest["layers"],
        "experts_per_layer": len(manifest["expert_ids"]),
        "tensor_bytes": tensor_bytes,
        "file_bytes": file_bytes,
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
