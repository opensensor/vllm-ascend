# SPDX-License-Identifier: Apache-2.0
"""Offline export -> checked child loading -> packed-weight CPU execution."""

import hashlib
import struct
import tempfile
from pathlib import Path

from .bundle import bundle_identity, canonical_json, export_plan, file_sha256
from .canonical_cpu import CanonicalCPUBackend
from .checkpoint import INDEX_NAME, read_header
from .placement import ASCEND_OWNERS, CUDA_OWNER, ExpertPlacement
from .reference import ExpertReply, combine, dispatch
from .runtime import LocalWorkerPool
from .wire import PacketLimits, WireIdentity
from .worker import DenseExpert, DenseExpertBank

FIXTURE_WIDTH = 32
FIXTURE_EXPERTS = 10
FIXTURE_TOKENS = 3
FIXTURE_TOP_K = 4
FIXTURE_STEPS = 3
FIXTURE_SHARD_TARGET = 2048


def _pack_codes(values, bits):
    fields_per_group, bytes_per_group = (8, 3) if bits == 3 else (8 // bits, 1)
    packed = bytearray()
    for start in range(0, len(values), fields_per_group):
        word = sum(
            (value & ((1 << bits) - 1)) << (bits * index)
            for index, value in enumerate(values[start : start + fields_per_group])
        )
        packed.extend(word.to_bytes(bytes_per_group, "little"))
    return bytes(packed)


def make_partition_fixture(source: Path):
    """Small synthetic source with real canonical packing and independent weights."""
    source.mkdir()
    owners = (CUDA_OWNER, *ASCEND_OWNERS)
    placement = ExpertPlacement(owners, (tuple(owners[expert % len(owners)] for expert in range(FIXTURE_EXPERTS)),))
    entries, experts = [], []
    for expert in range(FIXTURE_EXPERTS):
        matrices, bits = [], 2 + expert % 3
        for index, projection in enumerate(("gate", "up", "down")):
            signed = tuple(
                (row * 3 + column + expert + index) % (1 << bits) - (1 << (bits - 1))
                for row in range(FIXTURE_WIDTH)
                for column in range(FIXTURE_WIDTH)
            )
            scale = 1 / (256 if projection == "down" else 128)
            matrices.append(
                tuple(
                    tuple(value * scale for value in signed[row * FIXTURE_WIDTH : (row + 1) * FIXTURE_WIDTH])
                    for row in range(FIXTURE_WIDTH)
                )
            )
            prefix = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}_proj"
            entries.append(
                (prefix + "_codes", "U8", [FIXTURE_WIDTH, FIXTURE_WIDTH * bits // 8], _pack_codes(signed, bits))
            )
            scale_dtype, scale_format = ("F16", "<e") if expert % 2 else ("F32", "<f")
            entries.append((prefix + "_scale", scale_dtype, [1, 1], struct.pack(scale_format, scale)))
        experts.append((3, expert, DenseExpert(*matrices)))
    entries.append(("model.language_model.embed_tokens.weight", "F32", [1], struct.pack("<f", 0.25)))
    header, payload, records = {}, bytearray(), []
    for name, dtype, shape, data in entries:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw = canonical_json(header)
    raw += b" " * (-len(raw) % 8)
    filename = "fixture.safetensors"
    (source / filename).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    _, receipt = read_header(source / filename)
    weight_map = {name: filename for name, *_ in entries}
    (source / INDEX_NAME).write_bytes(canonical_json({"weight_map": weight_map}))
    (source / "config.json").write_bytes(
        canonical_json(
            {
                "model_type": "sidecard_cpu_fixture",
                "hidden_size": FIXTURE_WIDTH,
                "moe_intermediate_size": FIXTURE_WIDTH,
                "expert_scale_group": [32, 32],
                "precision": "unchanged_integer_codes_and_F16_or_F32_scales",
            }
        )
    )
    for name, dtype, shape, _ in entries:
        offsets = [value + receipt["payload_offset"] for value in header[name]["data_offsets"]]
        records.append({"name": name, "dtype": dtype, "shape": shape, "shard": filename, "file_offsets": offsets})
    partitions = {}
    for owner in owners:
        owned = [
            record
            for record in records
            if (
                placement.layer(3)[int(record["name"].split(".experts.")[1].split(".")[0])]
                if ".experts." in record["name"]
                else CUDA_OWNER
            )
            == owner
        ]
        partitions[owner] = {
            "tensor_count": len(owned),
            "source_tensor_bytes": sum(record["file_offsets"][1] - record["file_offsets"][0] for record in owned),
            "tensor_slices": owned,
        }
    plan = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_config_sha256": file_sha256(source / "config.json"),
        "source_index_sha256": file_sha256(source / INDEX_NAME),
        "placement_sha256": WireIdentity.for_placement(placement, "0" * 64).placement_sha256,
        "shards": {filename: receipt},
        "owners": partitions,
    }
    return placement, plan, DenseExpertBank(tuple(experts))


def run_partition_smoke():
    with tempfile.TemporaryDirectory(prefix="sidecard-partition-") as temporary:
        root = Path(temporary)
        placement, plan, serial_bank = make_partition_fixture(root / "source")
        source_digest = file_sha256(root / "source" / "fixture.safetensors")
        manifest = export_plan(plan, placement, root / "bundle", FIXTURE_SHARD_TARGET)
        identity = bundle_identity(root / "bundle", placement)
        limits = PacketLimits(
            FIXTURE_WIDTH, FIXTURE_TOKENS, FIXTURE_TOKENS * FIXTURE_TOP_K, FIXTURE_EXPERTS, FIXTURE_TOP_K
        )
        backends = {
            owner: CanonicalCPUBackend(root / "bundle", owner, placement, identity, FIXTURE_WIDTH)
            for owner in placement.owners
        }
        hidden = tuple(
            tuple((column % 7 - 3) / 16 * (token + 1) for column in range(FIXTURE_WIDTH))
            for token in range(FIXTURE_TOKENS)
        )
        ids = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 1))
        weights = ((0.125, 0.25, 0.25, 0.375),) * FIXTURE_TOKENS
        shared = ((0.03125,) * FIXTURE_WIDTH,) * FIXTURE_TOKENS
        replies = []
        for request in dispatch(placement, "serial", 3, hidden, ids, weights):
            outputs = tuple(serial_bank(3, route.expert, hidden[route.token]) for route in request.routes)
            replies.append(ExpertReply("serial", 3, request.owner, request.routes, outputs))
        expected = combine(placement, "serial", 3, hidden, ids, weights, replies, shared, 2.5)
        with LocalWorkerPool(placement, limits, identity, backends) as pool:
            pids = {owner: process.pid for owner, process in pool.processes.items()}
            results = [pool.execute(3, hidden, ids, weights, shared, 2.5) for _ in range(FIXTURE_STEPS)]
        if any(result != expected for result in results):
            raise AssertionError("exported packed-weight execution differs from independent source control")
        if file_sha256(root / "source" / "fixture.safetensors") != source_digest:
            raise AssertionError("export changed source checkpoint bytes")
        return {
            "status": "offline_partition_smoke_passed",
            "hardware_validated": False,
            "real_glm_weights": False,
            "native_layout_exported": False,
            "capacity_admitted": False,
            "matches_independent_source_control": True,
            "source_unchanged": True,
            "worker_preparation_complete": True,
            "worker_pids": pids,
            "steps": FIXTURE_STEPS,
            "hidden_width": FIXTURE_WIDTH,
            "expert_bits": [2, 3, 4],
            "scale_dtypes": ["F16", "F32"],
            "checkpoint_sha256": identity.checkpoint_sha256,
            "output_sha256": hashlib.sha256(canonical_json(expected)).hexdigest(),
            "owner_tensor_bytes": {owner: info["source_tensor_bytes"] for owner, info in manifest["owners"].items()},
            "owner_shards": {owner: len(info["shards"]) for owner, info in manifest["owners"].items()},
        }
