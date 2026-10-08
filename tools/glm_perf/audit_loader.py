# SPDX-License-Identifier: Apache-2.0
"""Audit GLM W2 loader coverage using checkpoint metadata only.

Run on the checkpoint host with ``python3 -m tools.glm_perf.audit_loader PATH``.
Only config.json, model.safetensors.index.json, and shard JSON headers are read.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import regex as re

INDEX_NAME = "model.safetensors.index.json"
EXPECTED_SHARDS = 33
EXPECTED_EXPERTS = 288
EXPECTED_DECODER_LAYERS = 42
EXPECTED_TP_SIZE = 4
EXPECTED_PEER_BYTES = 107_017_666_560
EXPECTED_LOCAL_BYTES = 35_672_555_520
EXPECTED_MTP_BYTES = 1_840_250_880
EXPECTED_EXTRA_ENTRIES = 10_368
EXPECTED_EXTRA_BYTES = 11_041_505_280
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
KINDS = {"codes": ("U8", 1), "scale": ("F32", 4)}
EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)_(codes|scale)$"
)
MAX_HEADER_BYTES = 64 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_header(path: Path) -> dict:
    """Read the eight-byte length and JSON header; never seek into payload."""
    with path.open("rb") as shard:
        prefix = shard.read(8)
        _require(len(prefix) == 8, f"{path}: truncated safetensors length")
        (header_length,) = struct.unpack("<Q", prefix)
        _require(0 < header_length <= MAX_HEADER_BYTES, f"{path}: invalid header length {header_length}")
        raw_header = shard.read(header_length)
        _require(len(raw_header) == header_length, f"{path}: truncated safetensors header")
    header = json.loads(raw_header)
    _require(isinstance(header, dict), f"{path}: header must be a JSON object")
    header.pop("__metadata__", None)
    return header


def _tensor_bytes(name: str, descriptor: dict, kind: str) -> int:
    dtype, bytes_per_element = KINDS[kind]
    _require(descriptor.get("dtype") == dtype, f"{name}: expected {dtype}, got {descriptor.get('dtype')}")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    _require(isinstance(shape, list) and all(isinstance(v, int) and v > 0 for v in shape), f"{name}: bad shape")
    _require(
        isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(v, int) for v in offsets),
        f"{name}: bad data offsets",
    )
    size = math.prod(shape) * bytes_per_element
    _require(offsets[0] >= 0 and offsets[1] - offsets[0] == size, f"{name}: offset/shape byte mismatch")
    return size


def _rank_range(rank: int, tp_size: int, experts: int) -> tuple[int, int]:
    # Mirror glm5next_w2.moe.ep_expert_range, including a short final block.
    per_rank = (experts + tp_size - 1) // tp_size
    start = min(rank * per_rank, experts)
    return start, min(start + per_rank, experts)


def _natural_sort_key(filename: str) -> list[str | int]:
    """Mirror vLLM's safetensors iterator file order."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)]


def audit_checkpoint(
    checkpoint: Path,
    *,
    expected_shards: int = EXPECTED_SHARDS,
    tp_size: int = EXPECTED_TP_SIZE,
    expected_experts: int = EXPECTED_EXPERTS,
    expected_decoder_layers: int = EXPECTED_DECODER_LAYERS,
    expected_mtp_layers: int = 1,
    expected_peer_bytes: int | None = EXPECTED_PEER_BYTES,
    expected_local_bytes: int | None = EXPECTED_LOCAL_BYTES,
    expected_mtp_bytes: int | None = EXPECTED_MTP_BYTES,
    expected_extra_entries: int | None = EXPECTED_EXTRA_ENTRIES,
    expected_extra_bytes: int | None = EXPECTED_EXTRA_BYTES,
) -> dict:
    """Check full expert coverage and all TP ranges from index/header metadata."""
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", {})
    _require(config.get("model_type") == "glm5_next" and text.get("model_type") == "glm5_next_text", "wrong GLM config")
    experts = text.get("n_routed_experts")
    layers = text.get("num_hidden_layers")
    dense = text.get("first_k_dense_replace")
    mtp_layers = text.get("num_nextn_predict_layers")
    _require(
        all(isinstance(v, int) for v in (experts, layers, dense, mtp_layers))
        and experts > 0
        and 0 <= dense < layers
        and mtp_layers >= 0
        and tp_size > 0,
        "invalid GLM expert/layer/TP geometry",
    )
    _require(layers - dense == expected_decoder_layers, "decoder layer count changed")
    _require(experts == expected_experts, "routed expert count changed")
    _require(mtp_layers == expected_mtp_layers, "MTP layer count changed")

    weight_map = json.loads((checkpoint / INDEX_NAME).read_text())["weight_map"]
    _require(isinstance(weight_map, dict) and weight_map, "empty safetensors weight map")
    indexed_shards = set(weight_map.values())
    actual_shards = {path.name for path in checkpoint.glob("*.safetensors")}
    _require(
        len(indexed_shards) == expected_shards,
        f"expected {expected_shards} indexed shards, got {len(indexed_shards)}",
    )
    _require(actual_shards == indexed_shards, "on-disk shards differ from the index")

    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        _require(isinstance(name, str) and isinstance(shard, str), "invalid weight-map entry")
        by_shard[shard].add(name)

    # The default iterator yields every header key in each selected *file*,
    # including old tensors superseded by an overlay in the weight map.
    # Each entry is (layer, expert, projection, kind, payload bytes, shard).
    indexed_experts: list[tuple[int, int, str, str, int, str]] = []
    stream_experts: list[tuple[int, int, str, str, int, str]] = []
    extra_entries = 0
    extra_bytes = 0
    indexed_bytes = 0
    last_source: dict[str, str] = {}
    sorted_shards = sorted(indexed_shards, key=_natural_sort_key)
    shard_order = {name: ordinal for ordinal, name in enumerate(sorted_shards)}
    for shard_name in sorted_shards:
        header = _read_header(checkpoint / shard_name)
        _require(by_shard[shard_name] <= set(header), f"{shard_name}: indexed tensor absent from header")
        for name, descriptor in header.items():
            offsets = descriptor.get("data_offsets")
            _require(isinstance(offsets, list) and len(offsets) == 2, f"{name}: bad data offsets")
            size = offsets[1] - offsets[0]
            is_indexed = weight_map.get(name) == shard_name
            if is_indexed:
                indexed_bytes += size
            else:
                _require(name in weight_map, f"{shard_name}: unindexed tensor {name}")
                _require(
                    shard_order[weight_map[name]] > shard_order[shard_name],
                    f"{shard_name}: superseded tensor {name} would overwrite indexed version",
                )
                extra_entries += 1
                extra_bytes += size
            last_source[name] = shard_name
            match = EXPERT_RE.fullmatch(name)
            if match is None:
                _require(is_indexed, f"{shard_name}: unexpected duplicate non-expert tensor {name}")
                _require(".mlp.experts." not in name, f"unrecognized expert tensor: {name}")
                continue
            layer, expert = int(match[1]), int(match[2])
            projection, kind = match[3], match[4]
            _require(0 <= expert < experts, f"{name}: expert ID outside config")
            _require(dense <= layer < layers + mtp_layers, f"{name}: expert in invalid layer")
            size = _tensor_bytes(name, descriptor, kind)
            entry = (layer, expert, projection, kind, size, shard_name)
            stream_experts.append(entry)
            if is_indexed:
                indexed_experts.append(entry)

    _require(last_source == weight_map, "final tensor source differs from safetensors index")
    _require(
        expected_extra_entries is None or extra_entries == expected_extra_entries,
        "extra header tensor count changed",
    )
    _require(expected_extra_bytes is None or extra_bytes == expected_extra_bytes, "extra header payload bytes changed")
    observed = {(layer, expert, projection, kind) for layer, expert, projection, kind, _, _ in indexed_experts}
    expected = {
        (layer, expert, projection, kind)
        for layer in range(dense, layers + mtp_layers)
        for expert in range(experts)
        for projection in PROJECTIONS
        for kind in KINDS
    }
    _require(
        observed == expected and len(indexed_experts) == len(expected),
        "expert tensor coverage is incomplete or duplicated",
    )
    mtp_entries = [entry for entry in stream_experts if entry[0] >= layers]
    mtp_bytes = sum(entry[4] for entry in mtp_entries)
    _require(expected_mtp_bytes is None or mtp_bytes == expected_mtp_bytes, "MTP bytes differ from checkpoint baseline")

    ranks = []
    for rank in range(tp_size):
        lo, hi = _rank_range(rank, tp_size, experts)
        decoder = [entry for entry in stream_experts if entry[0] < layers]
        local = [entry for entry in decoder if lo <= entry[1] < hi]
        peer = [entry for entry in decoder if not lo <= entry[1] < hi]
        local_bytes = sum(entry[4] for entry in local)
        peer_bytes = sum(entry[4] for entry in peer)
        _require(
            len({entry[:4] for entry in local}) == (layers - dense) * (hi - lo) * 6,
            f"rank {rank}: local tensor coverage mismatch",
        )
        _require(
            len({entry[:4] for entry in peer}) == (layers - dense) * (experts - hi + lo) * 6,
            f"rank {rank}: peer tensor coverage mismatch",
        )
        _require(
            expected_local_bytes is None or local_bytes == expected_local_bytes,
            f"rank {rank}: local bytes changed",
        )
        _require(
            expected_peer_bytes is None or peer_bytes == expected_peer_bytes,
            f"rank {rank}: skipped bytes changed",
        )
        retained_shards = {
            shard_name
            for name, shard_name in weight_map.items()
            if (match := EXPERT_RE.fullmatch(name)) is None or int(match[1]) >= layers or lo <= int(match[2]) < hi
        }
        _require(retained_shards == indexed_shards, f"rank {rank}: unexpected wholly skippable shard")
        ranks.append(
            {
                "rank": rank,
                "expert_range": [lo, hi],
                "local_decoder_tensors": len(local),
                "local_decoder_bytes": local_bytes,
                "skipped_peer_tensors": len(peer),
                "skipped_peer_bytes": peer_bytes,
                "retained_shards": len(retained_shards),
                "mtp_tensors_retained": len(mtp_entries),
                "mtp_bytes_retained": mtp_bytes,
            }
        )
    return {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "indexed_shards": len(indexed_shards),
        "indexed_tensors": len(weight_map),
        "indexed_payload_bytes": indexed_bytes,
        "extra_header_tensors": extra_entries,
        "extra_header_payload_bytes": extra_bytes,
        "stream_payload_bytes": indexed_bytes + extra_bytes,
        "final_tensor_sources_match_index": True,
        "decoder_layers": layers - dense,
        "mtp_layers": mtp_layers,
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_checkpoint(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
