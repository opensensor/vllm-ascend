# SPDX-License-Identifier: Apache-2.0
"""Audit local BF16 M3 headers and estimate a candidate W4 expert layout.

Only headers are read. Structural consistency is not payload integrity, model
quality, or runtime admission. The source checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

HEADER_LIMIT_BYTES = 16 << 20
EXPERT_BLOCK_SIZE = 32
FP16_BYTES = 2
INDEX_FILE = "model.safetensors.index.json"
EXPERT_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
)
LAYER_PATTERN = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        (length,) = struct.unpack("<Q", prefix)
        if not 0 < length <= HEADER_LIMIT_BYTES:
            raise ValueError(f"Invalid safetensors header length: {path}: {length}")
        raw = handle.read(length)
        if len(raw) != length:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(raw)
    header.pop("__metadata__", None)
    payload_bytes = path.stat().st_size - length - 8
    cursor = 0
    for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"]):
        dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4}.get(info["dtype"])
        if dtype_bytes is None:
            raise ValueError(f"Expected floating BF16/F16/F32 source, got {info['dtype']}: {name}")
        shape = info["shape"]
        if any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        start, end = info["data_offsets"]
        if start != cursor or end - start != math.prod(shape) * dtype_bytes or end > payload_bytes:
            raise ValueError(f"Invalid tensor offsets/size: {path}: {name}")
        cursor = end
    if cursor != payload_bytes:
        raise ValueError(f"Unaccounted payload bytes: {path}")
    return header, payload_bytes


def candidate_tensor_bytes(name: str, info: dict) -> int:
    """W4 experts with FP16 32x32 scales; BF16 becomes FP16; F32 stays F32."""
    count = math.prod(info["shape"])
    if EXPERT_PATTERN.fullmatch(name):
        if len(info["shape"]) != 2 or any(dim % EXPERT_BLOCK_SIZE for dim in info["shape"]):
            raise ValueError(f"Expert does not tile 32x32 blocks: {name}")
        return count // 2 + count // EXPERT_BLOCK_SIZE**2 * FP16_BYTES
    return count * (4 if info["dtype"] == "F32" else FP16_BYTES)


def audit_checkpoint(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    index_path = model_path / INDEX_FILE
    config = json.loads(config_path.read_text())
    if config.get("architectures") != ["MiniMaxM3SparseForConditionalGeneration"]:
        raise ValueError("Expected MiniMaxM3SparseForConditionalGeneration checkpoint")
    text = config["text_config"]
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    tensors = {}
    shards = []
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError(f"Expected a shard basename: {filename}")
        header, payload_bytes = read_header(model_path / filename)
        for name, info in header.items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            if weight_map.get(name) != filename:
                raise ValueError(f"Index/header mapping mismatch: {name}")
            tensors[name] = info
        shards.append({"name": filename, "payload_bytes": payload_bytes, "tensors": len(header)})
    missing = set(weight_map) - tensors.keys()
    if missing:
        raise ValueError(f"Indexed tensors absent from headers: {sorted(missing)[:5]}")

    layers = text["num_hidden_layers"]
    moe_layers = [layer for layer, enabled in enumerate(text["moe_layer_freq"]) if enabled]
    if len(text["moe_layer_freq"]) != layers:
        raise ValueError("MoE layer frequency length differs from num_hidden_layers")
    expected_experts = {
        (layer, expert, projection)
        for layer in moe_layers
        for expert in range(text["num_local_experts"])
        for projection in ("w1", "w2", "w3")
    }
    observed_experts = set()
    families = defaultdict(Counter)
    layer_bytes = [0] * layers
    first_stage_bytes = last_stage_bytes = 0
    for name, info in tensors.items():
        match = EXPERT_PATTERN.fullmatch(name)
        if match:
            layer, expert, projection = match.groups()
            observed_experts.add((int(layer), int(expert), projection))
            expected_shape = [text["intermediate_size"], text["hidden_size"]]
            if projection == "w2":
                expected_shape.reverse()
            if info["shape"] != expected_shape:
                raise ValueError(f"Expert geometry differs from config: {name}")
            family = "routed_experts"
        elif name.startswith("vision_tower."):
            family = "vision"
        else:
            family = "other"
        target_bytes = candidate_tensor_bytes(name, info)
        families[family].update(
            tensors=1,
            parameters=math.prod(info["shape"]),
            source_bytes=info["data_offsets"][1] - info["data_offsets"][0],
            candidate_bytes=target_bytes,
        )
        layer_match = LAYER_PATTERN.match(name)
        if layer_match:
            layer = int(layer_match[1])
            if layer >= layers:
                raise ValueError(f"Extra decoder/draft layer needs explicit accounting: {name}")
            layer_bytes[layer] += target_bytes
        elif name.startswith(("language_model.lm_head.", "language_model.model.norm.")):
            last_stage_bytes += target_bytes
        else:
            first_stage_bytes += target_bytes
    if observed_experts != expected_experts:
        raise ValueError("Routed expert coverage differs from config")

    payload_bytes = sum(shard["payload_bytes"] for shard in shards)
    warnings = []
    declared_bytes = index.get("metadata", {}).get("total_size")
    if declared_bytes != payload_bytes:
        warnings.append(f"Index total_size={declared_bytes}; actual tensor payload={payload_bytes}")
    warnings.append("Headers/lengths checked only; tensor payload hashes and upstream provenance are unverified")
    sparse = text["sparse_attention_config"]
    if len(sparse["sparse_attention_freq"]) != layers:
        raise ValueError("Sparse attention frequency length differs from num_hidden_layers")
    tp2_compatible = (
        all(
            text[key] % 2 == 0
            for key in ("num_attention_heads", "num_key_value_heads", "num_local_experts", "intermediate_size")
        )
        and sparse["sparse_num_index_heads"] % 2 == 0
    )
    # Stage totals are logical unique bytes, before TP replication or NZ padding.
    stage_candidates = []
    for first in range(1, layers - 1):
        for second in range(first + 1, layers):
            stage_bytes = [
                sum(layer_bytes[:first]) + first_stage_bytes,
                sum(layer_bytes[first:second]),
                sum(layer_bytes[second:]) + last_stage_bytes,
            ]
            stage_candidates.append((max(stage_bytes), first, second, stage_bytes))
    _, first, second, stage_bytes = min(stage_candidates)
    # Existing M3 backend retains full K/V despite sparse attention. Index K is
    # one index_dim vector/token replicated across the two TP workers per stage.
    sparse_layers = sum(bool(enabled) for enabled in sparse["sparse_attention_freq"])
    cache_bytes_per_token = (
        layers * 2 * text["num_key_value_heads"] * text["head_dim"] * FP16_BYTES
        + sparse_layers * sparse["sparse_index_dim"] * FP16_BYTES * 2
    )
    return {
        "schema_version": 1,
        "model_path": str(model_path.resolve()),
        "validation": "offline_header_structure_only",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "architectures": config["architectures"],
        "geometry": {
            key: text[key]
            for key in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "num_local_experts",
                "num_experts_per_tok",
                "hidden_act",
                "swiglu_alpha",
                "swiglu_limit",
                "max_position_embeddings",
            )
        },
        "sparse_attention_config": sparse,
        "mtp_config": {key: text.get(key) for key in ("num_nextn_predict_layers", "num_mtp_modules")},
        "source_payload_bytes": payload_bytes,
        "index_declared_bytes": declared_bytes,
        "tensor_count": len(tensors),
        "shards": shards,
        "families": dict(families),
        "candidate": {
            "format": "signed W4 experts, FP16 32x32 block scales; BF16->FP16; F32 preserved",
            "unique_weight_bytes": sum(item["candidate_bytes"] for item in families.values()),
            "exclusions": "TP replication, padding, KV/index cache, activations, graphs, communication, allocator",
            "tp6_query_heads_divisible": text["num_attention_heads"] % 6 == 0,
            "tp2_pp3_dimension_candidate": tp2_compatible,
            "byte_balanced_pp_layer_boundaries": [0, first, second, layers],
            "pp_stage_unique_bytes": stage_bytes,
            "tp2_fp16_cache_bytes_per_token_all_stages": cache_bytes_per_token,
            "mtp": "not established: config flags alone are insufficient; no extra decoder layers admitted",
        },
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_checkpoint(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("tensor_count", "source_payload_bytes", "candidate", "warnings")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
