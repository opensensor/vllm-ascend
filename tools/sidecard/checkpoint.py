# SPDX-License-Identifier: Apache-2.0
"""Plan canonical packed GLM tensor ownership using safetensors headers only.

This never loads tensor payloads or rewrites weights. Header validation is not
payload checksum validation, native layout export, or runtime memory admission.
"""

import hashlib
import json
import math
import re
import struct
from pathlib import Path

from .placement import (
    CUDA_OWNER,
    GLM_EXPERTS,
    GLM_FIRST_MOE_LAYER,
    GLM_HIDDEN,
    GLM_INTERMEDIATE,
    GLM_LAYERS,
    validate_glm_config,
)
from .wire import WireIdentity

MAX_HEADER_BYTES = 32 * 1024**2
INDEX_NAME = "model.safetensors.index.json"
EXPERT_TENSOR = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj_(codes|scale)$"
)
LAYER_TENSOR = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
PROJECTIONS = ("gate", "up", "down")
KINDS = ("codes", "scale")
BLOCK_SIZE = 32
DTYPE_BYTES = (
    ("BOOL", 1),
    ("U8", 1),
    ("I8", 1),
    ("F8_E4M3", 1),
    ("F8_E5M2", 1),
    ("I16", 2),
    ("U16", 2),
    ("F16", 2),
    ("BF16", 2),
    ("I32", 4),
    ("U32", 4),
    ("F32", 4),
    ("I64", 8),
    ("U64", 8),
    ("F64", 8),
)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON field")
        result[name] = value
    return result


def read_header(path: Path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= MAX_HEADER_BYTES or 8 + header_bytes > size:
            raise ValueError("unsafe safetensors header length")
        header_raw = handle.read(header_bytes)
    header = json.loads(header_raw, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    payload_start = 8 + header_bytes
    dtype_sizes = dict(DTYPE_BYTES)
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError("invalid safetensors tensor descriptor")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("invalid safetensors tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(value) is not int for value in offsets):
            raise ValueError("invalid safetensors tensor offsets")
        start, stop = offsets
        dtype_bytes = dtype_sizes.get(tensor.get("dtype"))
        if dtype_bytes is None or not 0 <= start <= stop <= size - payload_start:
            raise ValueError("unsupported dtype or out-of-file tensor slice")
        if stop - start != math.prod(shape) * dtype_bytes:
            raise ValueError("tensor dtype/shape disagrees with byte range")
        if stop > start:
            spans.append((start, stop))
    spans.sort()
    if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
        raise ValueError("overlapping safetensors tensor slices")
    receipt = {
        "file_bytes": size,
        "header_sha256": hashlib.sha256(header_raw).hexdigest(),
        "payload_offset": payload_start,
    }
    return header, receipt


def _expert_geometry(projection, kind, tensor, hidden=GLM_HIDDEN, intermediate=GLM_INTERMEDIATE):
    if any(type(value) is not int or value <= 0 or value % BLOCK_SIZE for value in (hidden, intermediate)):
        raise ValueError("canonical expert dimensions must contain complete block32 groups")
    rows, columns = (hidden, intermediate) if projection == "down" else (intermediate, hidden)
    if kind == "codes":
        bits = next((bits for bits in (2, 3, 4) if tensor["shape"] == [rows, columns * bits // 8]), None)
        if tensor["dtype"] != "U8" or bits is None:
            raise ValueError("expert codes require canonical W2/W3/W4 geometry")
        return bits
    if tensor["dtype"] not in ("F16", "F32") or tensor["shape"] != [rows // BLOCK_SIZE, columns // BLOCK_SIZE]:
        raise ValueError("expert scales require canonical block32 geometry")
    return None


def plan_checkpoint(source: Path, placement, include_tensor_slices=False):
    source = source.resolve()
    config_raw = (source / "config.json").read_bytes()
    config = json.loads(config_raw)
    validate_glm_config(config)
    if (
        config.get("ascend_glm_expert_layout") is not None
        or config.get("text_config", config).get("ascend_glm_expert_layout") is not None
    ):
        raise ValueError("native expert layout requires a dedicated native export adapter")
    if (
        CUDA_OWNER not in placement.owners
        or len(placement.layers[0]) != GLM_EXPERTS
        or len(placement.layers) != GLM_LAYERS - GLM_FIRST_MOE_LAYER
        or placement.first_moe_layer != GLM_FIRST_MOE_LAYER
    ):
        raise ValueError("checkpoint planner requires main-model GLM ownership including CUDA")
    index_raw = (source / INDEX_NAME).read_bytes()
    weight_map = json.loads(index_raw, object_pairs_hook=_unique_object)["weight_map"]
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(filename, str) for name, filename in weight_map.items()
    ):
        raise ValueError("invalid checkpoint weight map")
    headers, shards = {}, {}
    for filename in sorted(set(weight_map.values())):
        path = (source / filename).resolve()
        if not path.is_relative_to(source) or path == source:
            raise ValueError("checkpoint shard path escapes source directory")
        headers[filename], shards[filename] = read_header(path)
    owners = {
        owner: {
            "source_tensor_bytes": 0,
            "tensor_count": 0,
            "expert_tensor_count": 0,
            "layers": {},
            "tensor_slices": [],
        }
        for owner in placement.owners
    }
    seen, bits_by_layer = set(), {}
    excluded = {"vision": 0, "draft": 0}
    for name, filename in weight_map.items():
        tensor = headers[filename].get(name)
        if tensor is None:
            raise ValueError(f"indexed tensor is absent from shard header: {name}")
        layer_match = LAYER_TENSOR.search(name)
        if name.startswith("model.visual."):
            excluded["vision"] += 1
            continue
        if layer_match and int(layer_match[1]) >= placement.first_moe_layer + len(placement.layers):
            excluded["draft"] += 1
            continue
        match = EXPERT_TENSOR.fullmatch(name)
        owner = CUDA_OWNER
        if match:
            layer, expert = int(match[1]), int(match[2])
            ownership = placement.layer(layer)
            if not 0 <= expert < len(ownership):
                raise ValueError("checkpoint contains out-of-range expert")
            owner = ownership[expert]
            seen.add((layer, expert, match[3], match[4]))
            bits = _expert_geometry(match[3], match[4], tensor)
            if bits is not None:
                bits_by_layer.setdefault(str(layer), set()).add(bits)
            owners[owner]["expert_tensor_count"] += 1
        elif ".mlp.experts." in name:
            raise ValueError("unsupported expert tensor format; native bank export needs its own planner")
        record = owners[owner]
        start, stop = tensor["data_offsets"]
        record["source_tensor_bytes"] += stop - start
        record["tensor_count"] += 1
        if include_tensor_slices:
            record["tensor_slices"].append(
                {
                    "name": name,
                    "shard": filename,
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                    "file_offsets": [
                        start + shards[filename]["payload_offset"],
                        stop + shards[filename]["payload_offset"],
                    ],
                }
            )
    expected = {
        (layer, expert, projection, kind)
        for layer in range(GLM_FIRST_MOE_LAYER, GLM_LAYERS)
        for expert in range(GLM_EXPERTS)
        for projection in PROJECTIONS
        for kind in KINDS
    }
    if seen != expected:
        raise ValueError(f"incomplete expert coverage: {len(expected - seen)} missing tensors")
    for layer_id in range(GLM_FIRST_MOE_LAYER, GLM_LAYERS):
        for owner in placement.owners:
            # Position in this ordered list is the owner-local expert index.
            owners[owner]["layers"][str(layer_id)] = [
                expert for expert, assigned in enumerate(placement.layer(layer_id)) if assigned == owner
            ]
    if not include_tensor_slices:
        for record in owners.values():
            del record["tensor_slices"]
    return {
        "schema_version": 1,
        "status": "offline_canonical_tensor_plan",
        "source": str(source),
        "source_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "source_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "placement_sha256": WireIdentity.for_placement(placement, "0" * 64).placement_sha256,
        "shards": shards,
        "owners": owners,
        "expert_bits_by_layer": {layer: sorted(bits) for layer, bits in bits_by_layer.items()},
        "excluded_tensor_counts": excluded,
        "payload_checksums_verified": False,
        "native_layout_exported": False,
        "capacity_admitted": False,
        "hardware_validated": False,
        "mtp_enabled": False,
    }
