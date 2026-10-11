# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving export, corruption rejection, and verified worker preparation."""

import copy
import json
import socket
import struct
import subprocess
import sys
import time

import pytest

from tools.sidecard import bundle
from tools.sidecard.bundle import (
    BUNDLE_NAME,
    PartitionStore,
    bundle_identity,
    export_plan,
    file_sha256,
)
from tools.sidecard.canonical_cpu import CanonicalCPUBackend, decode_matrix
from tools.sidecard.partition_smoke import FIXTURE_WIDTH, make_partition_fixture
from tools.sidecard.placement import ExpertPlacement
from tools.sidecard.readiness import READY_MAGIC, WorkerStartupError, receive_ready, send_ready
from tools.sidecard.runtime import LocalWorkerPool
from tools.sidecard.smoke import smoke_fixture
from tools.sidecard.wire import PREFIX, WireIdentity
from tools.sidecard.worker import DenseExpertBank


@pytest.fixture
def partition(tmp_path):
    placement, plan, serial = make_partition_fixture(tmp_path / "source")
    root = tmp_path / "bundle"
    return placement, plan, serial, root


def test_export_preserves_every_tensor_byte_dtype_and_shape(partition):
    placement, plan, _, root = partition
    before = file_sha256(root.parent / "source" / "fixture.safetensors")
    manifest = export_plan(plan, placement, root, target_shard_bytes=800)
    identity = bundle_identity(root, placement)
    for owner, info in plan["owners"].items():
        store = PartitionStore(root, owner, placement, identity)
        assert len(store.records) == info["tensor_count"]
        for original in info["tensor_slices"]:
            record = store.records[original["name"]]
            with (root.parent / "source" / original["shard"]).open("rb") as handle:
                start, stop = original["file_offsets"]
                handle.seek(start)
                expected = handle.read(stop - start)
            assert store.read_tensor(record.name) == expected
            assert record.dtype == original["dtype"]
            assert record.shape == tuple(original["shape"])
        with pytest.raises(TypeError):
            store.records["new"] = None
    assert file_sha256(root.parent / "source" / "fixture.safetensors") == before
    assert manifest["export_payload_checksums_recorded"] is True
    assert manifest["source_payload_authenticated"] is False
    assert manifest["native_layout_exported"] is False
    assert manifest["capacity_admitted"] is False


def test_existing_destination_is_not_overwritten(partition):
    placement, plan, _, root = partition
    root.mkdir()
    (root / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        export_plan(plan, placement, root)
    assert (root / "sentinel").read_text() == "keep"


def test_export_identity_is_reproducible_across_destinations(partition):
    placement, plan, _, root = partition
    other = root.with_name("second-bundle")
    export_plan(plan, placement, root, target_shard_bytes=800)
    export_plan(plan, placement, other, target_shard_bytes=800)
    assert bundle_identity(root, placement) == bundle_identity(other, placement)


@pytest.mark.parametrize("failure", ["duplicate", "misowned", "slice", "missing", "header"])
def test_stale_or_incomplete_plan_rejected_before_destination_creation(partition, failure):
    placement, plan, _, root = partition
    plan = copy.deepcopy(plan)
    records = plan["owners"]["npu:0"]["tensor_slices"]
    if failure == "duplicate":
        records.append(records[0])
    elif failure == "misowned":
        records[0] = plan["owners"]["cuda:0"]["tensor_slices"][0]
    elif failure == "slice":
        records[0]["file_offsets"][0] += 1
    elif failure == "missing":
        records.pop()
    else:
        plan["shards"]["fixture.safetensors"]["header_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        export_plan(plan, placement, root)
    assert not root.exists()


def test_interrupted_export_never_publishes_complete_manifest(partition, monkeypatch):
    placement, plan, _, root = partition

    def fail(*args, **kwargs):
        raise OSError("injected copy failure")

    monkeypatch.setattr(bundle, "_copy_shard", fail)
    with pytest.raises(OSError):
        export_plan(plan, placement, root)
    assert not (root / BUNDLE_NAME).exists()
    with pytest.raises(FileNotFoundError):
        bundle_identity(root, placement)


def test_source_mutation_during_export_prevents_publication(partition, monkeypatch):
    placement, plan, _, root = partition
    original = bundle._copy_shard
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            with (root.parent / "source" / "fixture.safetensors").open("ab") as handle:
                handle.write(b"changed")
        return result

    monkeypatch.setattr(bundle, "_copy_shard", mutate)
    with pytest.raises(ValueError, match="changed during export"):
        export_plan(plan, placement, root)
    assert not (root / BUNDLE_NAME).exists()


@pytest.mark.parametrize("failure", ["shard", "index", "config", "placement", "manifest"])
def test_corrupted_partition_rejected_before_reads(partition, failure):
    placement, plan, _, root = partition
    manifest = export_plan(plan, placement, root)
    identity = bundle_identity(root, placement)
    info = manifest["owners"]["npu:0"]
    folder = root / info["folder"]
    if failure == "shard":
        filename = next(iter(info["shards"]))
        with (folder / filename).open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 1]))
    elif failure == "index":
        (folder / "model.safetensors.index.json").write_text("{}")
    elif failure == "config":
        (root / "source-config.json").write_text("{}")
    elif failure == "placement":
        (root / "placement.json").write_text("{}")
    else:
        (root / BUNDLE_NAME).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        PartitionStore(root, "npu:0", placement, identity)


@pytest.mark.parametrize("updates", [{"complete": False}, {"schema_version": True}, {"layout": "cube_n128_k256_v1"}])
def test_loader_rejects_unsupported_bundle_even_with_recomputed_identity(partition, updates):
    placement, plan, _, root = partition
    manifest = export_plan(plan, placement, root)
    manifest.update(updates)
    (root / BUNDLE_NAME).write_bytes(bundle.canonical_json(manifest))
    with pytest.raises(ValueError, match="unsupported sidecard bundle"):
        PartitionStore(root, "npu:0", placement, bundle_identity(root, placement))


def test_verify_cli_reports_pinned_partition_and_rejects_wrong_pin(partition):
    placement, plan, _, root = partition
    export_plan(plan, placement, root)
    identity = bundle_identity(root, placement)
    command = [
        sys.executable,
        "-m",
        "tools.sidecard",
        "verify-partition",
        "--root",
        str(root),
        "--placement",
        str(root / "placement.json"),
        "--owner",
        "npu:0",
        "--expected-checkpoint-sha256",
    ]
    result = subprocess.run([*command, identity.checkpoint_sha256], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["manifest_pinned"] is True
    assert report["shard_checksums_verified"] is True
    assert report["hardware_validated"] is False
    failed = subprocess.run([*command, "0" * 64], capture_output=True, text=True)
    assert failed.returncode != 0
    assert "pinned checkpoint" in failed.stderr


def test_streamed_read_detects_post_verification_mutation_and_preserves_old_bytes(partition):
    placement, plan, _, root = partition
    export_plan(plan, placement, root)
    store = PartitionStore(root, "npu:0", placement, bundle_identity(root, placement))
    record = next(iter(store.records.values()))
    retained = store.read_tensor(record.name)
    assert b"".join(store.iter_chunks(record.name, chunk_bytes=3)) == retained
    with pytest.raises(ValueError, match="budget"):
        store.read_tensor(record.name, max_bytes=1)
    with record.path.open("r+b") as handle:
        handle.seek(record.file_offsets[0])
        handle.write(bytes([retained[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        store.read_tensor(record.name)
    with record.path.open("rb") as handle:
        handle.seek(record.file_offsets[0])
        assert retained[0] != handle.read(1)[0]


def test_empty_owner_partition_is_valid_and_contains_no_peer_weights(partition):
    placement, plan, _, root = partition
    placement = ExpertPlacement(placement.owners, (("cuda:0",) * len(placement.layer(3)),))
    plan["placement_sha256"] = WireIdentity.for_placement(placement, "0" * 64).placement_sha256
    all_records = [record for info in plan["owners"].values() for record in info["tensor_slices"]]
    for owner, info in plan["owners"].items():
        info["tensor_slices"] = all_records if owner == "cuda:0" else []
        info["tensor_count"] = len(info["tensor_slices"])
        info["source_tensor_bytes"] = sum(
            record["file_offsets"][1] - record["file_offsets"][0] for record in info["tensor_slices"]
        )
    export_plan(plan, placement, root)
    store = PartitionStore(root, "npu:0", placement, bundle_identity(root, placement))
    assert not store.records
    assert store.source_tensor_bytes == 0


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("dtype", ["F16", "F32"])
def test_canonical_signed_field_decode(bits, dtype):
    # Byte patterns independently specify the expected two's-complement sequence.
    patterns = {
        2: (b"\x4e", (-2, -1, 0, 1)),
        3: (bytes.fromhex("ac8f68"), (-4, -3, -2, -1, 0, 1, 2, 3)),
        4: (b"\xf8\x70", (-8, -1, 0, 7)),
    }
    packed, values = patterns[bits]
    count = FIXTURE_WIDTH * FIXTURE_WIDTH // len(values)
    scales = struct.pack("<e" if dtype == "F16" else "<f", 0.25)
    matrix = decode_matrix(packed * count, scales, FIXTURE_WIDTH, FIXTURE_WIDTH, bits, dtype)
    assert tuple(value for row in matrix for value in row) == tuple(value * 0.25 for value in values) * count


def test_checked_cpu_backend_matches_independent_source_and_enforces_budget(partition):
    placement, plan, serial, root = partition
    export_plan(plan, placement, root)
    identity = bundle_identity(root, placement)
    backend = CanonicalCPUBackend(root, "npu:0", placement, identity, FIXTURE_WIDTH)
    with pytest.raises(RuntimeError, match="preparation"):
        backend.compute(None)
    assert backend.prepare() == ((3, 1), (3, 6))
    hidden = (0.0625,) * FIXTURE_WIDTH
    assert backend.bank(3, 1, hidden) == serial(3, 1, hidden)
    with pytest.raises(ValueError, match="budget"):
        CanonicalCPUBackend(root, "npu:0", placement, identity, FIXTURE_WIDTH, element_limit=1).prepare()
    with pytest.raises(ValueError, match="dimensions"):
        CanonicalCPUBackend(root, "npu:0", placement, identity, 4096)
    with pytest.raises(ValueError, match="hidden width"):
        CanonicalCPUBackend(root, "npu:0", placement, identity, 64).prepare()


def test_failed_repreparation_does_not_reuse_stale_cpu_weights(partition):
    placement, plan, _, root = partition
    manifest = export_plan(plan, placement, root)
    backend = CanonicalCPUBackend(root, "npu:0", placement, bundle_identity(root, placement), FIXTURE_WIDTH)
    backend.prepare()
    assert backend.bank is not None
    info = manifest["owners"]["npu:0"]
    path = root / info["folder"] / next(iter(info["shards"]))
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        backend.prepare()
    assert backend.bank is None
    with pytest.raises(RuntimeError, match="preparation"):
        backend.compute(None)


def test_loader_rejects_metadata_symlink_outside_bundle(partition):
    placement, plan, _, root = partition
    export_plan(plan, placement, root)
    identity = bundle_identity(root, placement)
    outside = root.parent / "outside-config.json"
    outside.write_bytes((root / "source-config.json").read_bytes())
    (root / "source-config.json").unlink()
    (root / "source-config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        PartitionStore(root, "npu:0", placement, identity)


def test_readiness_checks_owner_and_session_identity():
    placement, _, identity, _, _ = smoke_fixture()
    left, right = socket.socketpair()
    try:
        send_ready(left, placement.owners[0], identity, 1)
        with pytest.raises(ValueError, match="identity"):
            receive_ready(right, placement.owners[1], identity, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


def test_readiness_rejects_body_allocation():
    _, _, identity, _, _ = smoke_fixture()
    left, right = socket.socketpair()
    try:
        left.sendall(PREFIX.pack(READY_MAGIC, 1, 1))
        with pytest.raises(ValueError, match="readiness frame"):
            receive_ready(right, "cuda:0", identity, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


def test_readiness_surfaces_bounded_backend_error():
    _, _, identity, _, _ = smoke_fixture()
    left, right = socket.socketpair()
    try:
        send_ready(left, "npu:0", identity, 1, error="checksum mismatch " * 100)
        with pytest.raises(WorkerStartupError, match="checksum mismatch") as error:
            receive_ready(right, "npu:0", identity, time.monotonic() + 1)
        assert len(str(error.value)) < 200
    finally:
        left.close()
        right.close()


def test_pool_rejects_false_ready_backend_with_missing_experts():
    placement, limits, identity, backends, _ = smoke_fixture()
    backends["npu:0"] = DenseExpertBank(())
    with pytest.raises(WorkerStartupError, match="resident expert coverage"):
        LocalWorkerPool(placement, limits, identity, backends)


def test_export_load_process_cli_matches_independent_packed_weight_control():
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard", "partition-smoke"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    report = json.loads(result.stdout)
    assert report["matches_independent_source_control"] is True
    assert report["source_unchanged"] is True
    assert report["worker_preparation_complete"] is True
    assert report["hardware_validated"] is False
    assert report["native_layout_exported"] is False
    assert report["real_glm_weights"] is False
    assert report["expert_bits"] == [2, 3, 4]
    assert len(set(report["worker_pids"].values())) == 5
    assert report["steps"] == 3
