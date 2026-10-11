# SPDX-License-Identifier: Apache-2.0
"""Stream unchanged owner tensors into checked partitions; publish metadata last."""

import hashlib
import json
import os
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .checkpoint import (
    EXPERT_TENSOR,
    INDEX_NAME,
    LAYER_TENSOR,
    _expert_geometry,
    _unique_object,
    plan_checkpoint,
    read_header,
)
from .placement import CUDA_OWNER, ExpertPlacement
from .wire import WireIdentity

BUNDLE_VERSION = 1
BUNDLE_NAME = "sidecard-bundle.json"
CANONICAL_LAYOUT = "canonical_block32_v1"
COPY_CHUNK_BYTES = 8 * 1024**2
DEFAULT_SHARD_BYTES = 1024**3
DEFAULT_TENSOR_READ_BYTES = 64 * 1024**2


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _write_metadata(path, raw):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inside(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("invalid bundle relative path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("bundle path escapes its directory")
    return path


def assigned_owner(name, placement):
    """Dense text weights belong to CUDA; excluded state never reaches workers."""
    if name.startswith("model.visual."):
        return None
    layer_match = LAYER_TENSOR.search(name)
    if layer_match and int(layer_match[1]) >= placement.first_moe_layer + len(placement.layers):
        return None
    match = EXPERT_TENSOR.fullmatch(name)
    if match:
        expert = int(match[2])
        layer = placement.layer(int(match[1]))
        if not 0 <= expert < len(layer):
            raise ValueError("out-of-range expert in partition")
        return layer[expert]
    if ".mlp.experts." in name:
        raise ValueError("unsupported expert tensor naming")
    return CUDA_OWNER


def _canonical_config(raw):
    config = json.loads(raw, object_pairs_hook=_unique_object)
    text = config.get("text_config", config)
    if config.get("ascend_glm_expert_layout") is not None or text.get("ascend_glm_expert_layout") is not None:
        raise ValueError("native layout cannot be labeled as canonical")
    return text["hidden_size"], text["moe_intermediate_size"]


def _check_plan(plan, placement):
    if plan.get("placement_sha256") != WireIdentity.for_placement(placement, "0" * 64).placement_sha256:
        raise ValueError("export plan placement mismatch")
    if set(plan["owners"]) != set(placement.owners):
        raise ValueError("export plan owner coverage mismatch")
    source = Path(plan["source"]).resolve()
    if (
        file_sha256(source / INDEX_NAME) != plan["source_index_sha256"]
        or file_sha256(source / "config.json") != plan["source_config_sha256"]
    ):
        raise ValueError("source config or index changed after planning")
    index = json.loads((source / INDEX_NAME).read_text(), object_pairs_hook=_unique_object)["weight_map"]
    hidden, intermediate = _canonical_config((source / "config.json").read_bytes())
    headers, snapshots = {}, {}
    for filename, receipt in plan["shards"].items():
        path = _inside(source, filename)
        snapshots[filename] = _snapshot(path)
        headers[filename], actual = read_header(path)
        if actual != receipt:
            raise ValueError("source shard header changed after planning")
    seen = set()
    for owner, info in plan["owners"].items():
        records = info.get("tensor_slices")
        if records is None:
            raise ValueError("export plan requires tensor slices")
        for record in records:
            name, filename = record["name"], record["shard"]
            if name in seen or assigned_owner(name, placement) != owner or index.get(name) != filename:
                raise ValueError("duplicate, missing or misowned source tensor")
            tensor = headers[filename][name]
            match = EXPERT_TENSOR.fullmatch(name)
            if match:
                _expert_geometry(match[3], match[4], tensor, hidden, intermediate)
            base = plan["shards"][filename]["payload_offset"]
            if (record["dtype"], record["shape"], record["file_offsets"]) != (
                tensor["dtype"],
                tensor["shape"],
                [offset + base for offset in tensor["data_offsets"]],
            ):
                raise ValueError("source tensor slice differs from its header")
            seen.add(name)
        if info["tensor_count"] != len(records) or info["source_tensor_bytes"] != sum(
            record["file_offsets"][1] - record["file_offsets"][0] for record in records
        ):
            raise ValueError("export plan counts or bytes disagree")
    if seen != {name for name in index if assigned_owner(name, placement) is not None}:
        raise ValueError("export plan lacks complete tensor coverage")
    return source, snapshots


def _groups(records, target_bytes):
    group, size = [], 0
    for record in sorted(records, key=lambda value: (value["shard"], value["file_offsets"][0])):
        count = record["file_offsets"][1] - record["file_offsets"][0]
        if group and size + count > target_bytes:
            yield group
            group, size = [], 0
        group.append(record)
        size += count
    if group:
        yield group


def _copy_shard(path, records, handles):
    header, offset = {"__metadata__": {"format": "pt", "sidecard_layout": CANONICAL_LAYOUT}}, 0
    for record in records:
        count = record["file_offsets"][1] - record["file_offsets"][0]
        header[record["name"]] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "data_offsets": [offset, offset + count],
        }
        offset += count
    raw = canonical_json(header)
    # Safetensors headers are padded to eight bytes; tensor bytes are unchanged.
    raw += b" " * (-len(raw) % 8)
    hashes = {}
    expected_digest = hashlib.sha256(struct.pack("<Q", len(raw)) + raw)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for record in records:
            source = handles[record["shard"]]
            start, stop = record["file_offsets"]
            source.seek(start)
            remaining, digest = stop - start, hashlib.sha256()
            while remaining:
                chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("source tensor truncated during export")
                output.write(chunk)
                expected_digest.update(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            hashes[record["name"]] = digest.hexdigest()
        output.flush()
        os.fsync(output.fileno())
    read_header(path)
    actual_digest = file_sha256(path)
    if actual_digest != expected_digest.hexdigest():
        raise ValueError("exported shard differs from copied source bytes")
    return {"file_bytes": path.stat().st_size, "sha256": actual_digest}, hashes


def export_plan(plan, placement, destination: Path, target_shard_bytes=DEFAULT_SHARD_BYTES):
    """Generic slice copier; production GLM calls export_checkpoint for planning.

    New destinations only. Failures leave an incomplete directory without a
    complete manifest; no existing checkpoint or partition is overwritten.
    """
    if type(target_shard_bytes) is not int or target_shard_bytes <= 0:
        raise ValueError("shard target must be a positive byte count")
    source, snapshots = _check_plan(plan, placement)
    config_raw = (source / "config.json").read_bytes()
    if hashlib.sha256(config_raw).hexdigest() != plan["source_config_sha256"]:
        raise ValueError("source config changed during export preparation")
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=False)
    owner_records = {}
    with ExitStack() as stack:
        handles = {name: stack.enter_context(_inside(source, name).open("rb")) for name in snapshots}
        for number, owner in enumerate(placement.owners):
            folder = f"worker-{number:04d}"
            root = destination / folder
            root.mkdir()
            index, shards, tensor_hashes = {}, {}, {}
            info = plan["owners"][owner]
            for shard_number, records in enumerate(_groups(info["tensor_slices"], target_shard_bytes)):
                filename = f"weights-{shard_number:05d}.safetensors"
                shards[filename], hashes = _copy_shard(root / filename, records, handles)
                tensor_hashes.update(hashes)
                index.update({record["name"]: filename for record in records})
            raw_index = canonical_json({"metadata": {"total_size": info["source_tensor_bytes"]}, "weight_map": index})
            _write_metadata(root / INDEX_NAME, raw_index)
            owner_records[owner] = {
                "folder": folder,
                "source_tensor_bytes": info["source_tensor_bytes"],
                "tensor_count": len(index),
                "shards": shards,
                "index_sha256": hashlib.sha256(raw_index).hexdigest(),
                "tensor_sha256": tensor_hashes,
            }
        if any(_snapshot(_inside(source, name)) != before for name, before in snapshots.items()):
            raise ValueError("source shard changed during export")
        if (
            file_sha256(source / INDEX_NAME) != plan["source_index_sha256"]
            or file_sha256(source / "config.json") != plan["source_config_sha256"]
        ):
            raise ValueError("source metadata changed during export")
    _write_metadata(destination / "source-config.json", config_raw)
    _write_metadata(destination / "placement.json", canonical_json(placement.to_dict()))
    manifest = {
        "schema_version": BUNDLE_VERSION,
        "complete": True,
        "layout": CANONICAL_LAYOUT,
        "placement_sha256": plan["placement_sha256"],
        "source_config_sha256": plan["source_config_sha256"],
        "source_index_sha256": plan["source_index_sha256"],
        "source_payload_authenticated": False,
        "export_payload_checksums_recorded": True,
        "owners": owner_records,
        "native_layout_exported": False,
        "capacity_admitted": False,
        "hardware_validated": False,
        "mtp_enabled": False,
    }
    # Publish only after every shard is durable and its output checksum recorded.
    _write_metadata(destination / BUNDLE_NAME, canonical_json(manifest))
    return manifest


def export_checkpoint(source, placement, destination, target_shard_bytes=DEFAULT_SHARD_BYTES):
    return export_plan(plan_checkpoint(source, placement, True), placement, destination, target_shard_bytes)


def bundle_identity(root: Path, placement: ExpertPlacement):
    return WireIdentity.for_placement(placement, file_sha256(_inside(root, BUNDLE_NAME)))


@dataclass(frozen=True)
class TensorRecord:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    file_offsets: tuple[int, int]
    sha256: str


class PartitionStore:
    """Verify a single owner before exposing bounded, privately owned byte reads."""

    def __init__(self, root: Path, owner: str, placement: ExpertPlacement, identity: WireIdentity):
        root = root.resolve()
        # Pin deployment metadata before reading any owner payloads.
        if bundle_identity(root, placement) != identity:
            raise ValueError("bundle identity differs from configured session")
        manifest = json.loads(_inside(root, BUNDLE_NAME).read_text(), object_pairs_hook=_unique_object)
        if (
            type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != BUNDLE_VERSION
            or manifest.get("complete") is not True
            or manifest.get("layout") != CANONICAL_LAYOUT
        ):
            raise ValueError("incomplete or unsupported sidecard bundle")
        if (
            manifest["placement_sha256"] != identity.placement_sha256
            or ExpertPlacement.from_dict(json.loads(_inside(root, "placement.json").read_text())) != placement
        ):
            raise ValueError("bundle placement mismatch")
        if file_sha256(_inside(root, "source-config.json")) != manifest["source_config_sha256"] or set(
            manifest["owners"]
        ) != set(placement.owners):
            raise ValueError("bundle config or owner coverage mismatch")
        hidden, intermediate = _canonical_config(_inside(root, "source-config.json").read_bytes())
        if owner not in manifest["owners"]:
            raise ValueError("unknown owner partition")
        info = manifest["owners"][owner]
        folder = _inside(root, info["folder"])
        index_path = _inside(folder, INDEX_NAME)
        if file_sha256(index_path) != info["index_sha256"]:
            raise ValueError("partition index checksum mismatch")
        index = json.loads(index_path.read_text(), object_pairs_hook=_unique_object)["weight_map"]
        if (
            set(index) != set(info["tensor_sha256"])
            or len(index) != info["tensor_count"]
            or set(index.values()) != set(info["shards"])
        ):
            raise ValueError("partition tensor/shard coverage mismatch")
        headers, offsets = {}, {}
        # Verify complete shard bytes before exposing immutable tensor slices.
        for filename, receipt in info["shards"].items():
            path = _inside(folder, filename)
            if path.stat().st_size != receipt["file_bytes"] or file_sha256(path) != receipt["sha256"]:
                raise ValueError("partition shard checksum mismatch")
            headers[filename], record = read_header(path)
            offsets[filename] = record["payload_offset"]
        records = {}
        for name, filename in index.items():
            if assigned_owner(name, placement) != owner or name not in headers[filename]:
                raise ValueError("foreign or missing tensor in partition")
            tensor = headers[filename][name]
            match = EXPERT_TENSOR.fullmatch(name)
            if match:
                _expert_geometry(match[3], match[4], tensor, hidden, intermediate)
            records[name] = TensorRecord(
                name,
                _inside(folder, filename),
                tensor["dtype"],
                tuple(tensor["shape"]),
                tuple(value + offsets[filename] for value in tensor["data_offsets"]),
                info["tensor_sha256"][name],
            )
        required = {
            f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj_{kind}"
            for layer in range(placement.first_moe_layer, placement.first_moe_layer + len(placement.layers))
            for expert, assigned in enumerate(placement.layer(layer))
            if assigned == owner
            for projection in ("gate", "up", "down")
            for kind in ("codes", "scale")
        }
        # Source names remain global; coverage must exactly match this owner.
        if {name for name in records if EXPERT_TENSOR.fullmatch(name)} != required:
            raise ValueError("incomplete local expert coverage")
        if (
            sum(record.file_offsets[1] - record.file_offsets[0] for record in records.values())
            != info["source_tensor_bytes"]
        ):
            raise ValueError("partition byte count mismatch")
        self.owner, self.placement, self.identity = owner, placement, identity
        self.records = MappingProxyType(records)
        self.source_tensor_bytes = info["source_tensor_bytes"]
        self.hidden_width, self.intermediate_size = hidden, intermediate

    def iter_chunks(self, name, chunk_bytes=COPY_CHUNK_BYTES):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= COPY_CHUNK_BYTES:
            raise ValueError("invalid tensor chunk size")
        record = self.records[name]
        start, stop = record.file_offsets
        digest = hashlib.sha256()
        with record.path.open("rb") as handle:
            handle.seek(start)
            remaining = stop - start
            while remaining:
                data = handle.read(min(remaining, chunk_bytes))
                if not data:
                    raise ValueError("partition tensor truncated during read")
                remaining -= len(data)
                digest.update(data)
                yield data
        if digest.hexdigest() != record.sha256:
            raise ValueError("partition tensor checksum mismatch")

    def read_tensor(self, name, max_bytes=DEFAULT_TENSOR_READ_BYTES):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("tensor read budget must be positive")
        record = self.records[name]
        if record.file_offsets[1] - record.file_offsets[0] > max_bytes:
            raise ValueError("tensor exceeds read budget; use streamed staging")
        return b"".join(self.iter_chunks(name))
