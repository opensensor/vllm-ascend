# SPDX-License-Identifier: Apache-2.0
"""Verify a self-contained card-profile checkpoint offline; never runs inference."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from .deployment_profiles import validate_profile
from .fused_weight_layout import FP16_SCALE_LAYOUT
from .indexer_bundle import verified_manifest
from .native_checkpoint import INDEX, LAYOUT, MANIFEST, file_digest, kernel_assets, validate_scale_contract


def verify(folder, *, verify_sha=True):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / MANIFEST).read_text())
    config = json.loads((folder / "config.json").read_text())
    index = json.loads((folder / INDEX).read_text())["weight_map"]
    if manifest.get("complete") is not True or manifest.get("layout") != LAYOUT:
        raise ValueError("checkpoint is incomplete or has an unsupported layout")
    if config.get("ascend_glm_expert_layout") != LAYOUT:
        raise ValueError("checkpoint layout markers disagree")
    for entry in manifest.get("source_attribution", {}).get("files", []):
        path = folder / entry["file"]
        if not path.resolve().is_relative_to(folder) or file_digest(path) != entry["sha256"]:
            raise ValueError("source attribution differs from manifest")
    for field in ("kernel_bundle", "decode_kernel_bundle"):
        if field not in manifest:
            continue
        root = folder / manifest[field]
        if not root.resolve().is_relative_to(folder):
            raise ValueError("kernel bundle escapes checkpoint")
        provenance, _ = kernel_assets(root)
        digest_key = "kernel_provenance_sha256" if field == "kernel_bundle" else "decode_kernel_provenance_sha256"
        if file_digest(root / "provenance.json") != manifest[digest_key]:
            raise ValueError("kernel provenance differs from manifest")
        validate_scale_contract(manifest, config.get("ascend_glm_expert_scale_layout"), provenance["_build"])
    name = manifest.get("deployment_profile")
    if "indexer_kernel_bundle" in manifest:
        root = folder / manifest["indexer_kernel_bundle"]
        if not root.resolve().is_relative_to(folder):
            raise ValueError("indexer bundle escapes checkpoint")
        if file_digest(root / "manifest.json") != manifest["indexer_manifest_sha256"]:
            raise ValueError("indexer manifest differs from checkpoint")
        verified_manifest(root)
    if config.get("ascend_glm_deployment_profile") != name:
        raise ValueError("checkpoint deployment profile markers disagree")
    if name is not None:
        topology = {"two-card-selective-w2": (4, 1, 1), "three-card-w3": (2, 3, 0)}.get(name)
        if topology is None:
            raise ValueError("unknown deployment profile")
        tp, pp, mtp = topology
        validate_profile(
            name, tensor_parallel_size=tp, pipeline_parallel_size=pp, mtp_tokens=mtp, layer_bits=manifest["layer_bits"]
        )
        if manifest["world_size"] != tp:
            raise ValueError("expert partition differs from deployment profile")
    records = {}
    for record in manifest["native_shards"] + manifest["scale_shards"] + manifest["dense_shards"]:
        filename = record["file"]
        if filename in records and records[filename] != record:
            raise ValueError("conflicting shard records")
        records[filename] = record
    if set(index.values()) != records.keys():
        raise ValueError("index and manifest shard coverage differ")
    if {p.name for p in folder.glob("*.safetensors")} != records.keys():
        raise ValueError("checkpoint contains missing or unindexed tensor shards")
    by_file = {filename: set() for filename in records}
    for name, filename in index.items():
        by_file[filename].add(name)
    codes, scales = 0, 0
    for filename, record in records.items():
        path = folder / filename
        if not path.resolve().is_relative_to(folder) or path.stat().st_size != record["bytes"]:
            raise ValueError("shard path or byte count differs from manifest")
        if verify_sha and file_digest(path) != record["sha256"]:
            raise ValueError("shard checksum differs from manifest")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = handle.keys()
            if set(names) != by_file[filename]:
                raise ValueError("shard contains missing or unindexed tensors")
            for name in names:
                header = handle.get_slice(name)
                if ".mlp.experts." not in name:
                    continue
                if name.endswith("_proj_codes"):
                    if header.get_dtype() != "U8" or (handle.metadata() or {}).get("layout") != LAYOUT:
                        raise ValueError("expert code dtype or layout differs")
                    codes += 1
                elif name.endswith("_proj_scale"):
                    if (
                        header.get_dtype() != "F16"
                        or (handle.metadata() or {}).get("weight_scale_layout") != FP16_SCALE_LAYOUT
                    ):
                        raise ValueError("expert scale dtype or layout differs")
                    scales += 1
    expected = 3 * manifest["num_experts"] * len(manifest["layers"])
    if codes != expected or scales != expected:
        raise ValueError("expert tensor coverage differs from geometry")
    return dict(
        profile=manifest.get("deployment_profile"),
        world_size=manifest["world_size"],
        shard_count=len(records),
        tensor_count=len(index),
        code_tensors=codes,
        scale_tensors=scales,
        tensor_bytes=sum(r["bytes"] for r in records.values()),
        shard_checksums_verified=verify_sha,
        validation="Offline storage checks only; hardware, quality and performance remain unqualified",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--headers-only", action="store_true", help="Skip tensor payload SHA checks")
    args = parser.parse_args()
    report = verify(args.model, verify_sha=not args.headers_only)
    value = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(value)
    print(value, end="")


if __name__ == "__main__":
    main()
