# SPDX-License-Identifier: Apache-2.0
"""Separate interpreters, readiness, fail-closed exits and explicit route maps."""

import json
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools.sidecard.bundle import bundle_identity, export_plan
from tools.sidecard.canonical_cpu import CanonicalCPUBackend
from tools.sidecard.cuda_combine import pack_float_rows
from tools.sidecard.device_backend import NativeGateBackend, compact_routes
from tools.sidecard.external_runtime import ExternalWorkerPool, WorkerLaunch
from tools.sidecard.external_worker import worker_from_config
from tools.sidecard.partition_smoke import make_partition_fixture
from tools.sidecard.placement import glm_placement
from tools.sidecard.readiness import WorkerStartupError
from tools.sidecard.reference import ExpertRequest, Route
from tools.sidecard.runtime import LocalWorkerPool
from tools.sidecard.split_gate import compare_rows, gate_placement, read_checked
from tools.sidecard.wire import PREFIX, PacketLimits, WireIdentity


@pytest.fixture
def external_fixture(tmp_path):
    placement, plan, _ = make_partition_fixture(tmp_path / "source")
    export_plan(plan, placement, tmp_path / "bundle")
    identity = bundle_identity(tmp_path / "bundle", placement)
    limits = PacketLimits(32, 3, 12, 10, 4)
    launches = {
        owner: WorkerLaunch(
            Path(sys.executable), {"kind": "canonical_cpu", "root": str(tmp_path / "bundle"), "hidden_width": 32}
        )
        for owner in placement.owners
    }
    return placement, identity, limits, launches


def test_external_workers_match_spawned_reference_and_exit_without_termination(tmp_path, external_fixture):
    placement, identity, limits, launches = external_fixture
    hidden = ((0.25,) * 32,) * 3
    ids = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 1))
    weights = ((0.0, 0.25, 0.375, 0.375),) * 3
    shared = ((0.03125,) * 32,) * 3
    backends = {
        owner: CanonicalCPUBackend(tmp_path / "bundle", owner, placement, identity, 32) for owner in placement.owners
    }
    with LocalWorkerPool(placement, limits, identity, backends) as pool:
        expected = pool.execute(3, hidden, ids, weights, shared, 2.5)
    with ExternalWorkerPool(placement, limits, identity, launches, tmp_path / "logs") as pool:
        processes = list(pool.processes.values())
        first = pool.execute(3, hidden, ids, weights, shared, 2.5)
        second = pool.execute(3, hidden, ids, weights, shared, 2.5)
        assert first == expected == second
    assert all(process.poll() == 0 for process in processes)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.execute(3, hidden, ids, weights, shared, 2.5)


def test_external_startup_failure_surfaces_and_releases_other_workers(tmp_path, external_fixture):
    placement, identity, limits, launches = external_fixture
    launches[placement.owners[0]].backend["root"] = str(tmp_path / "missing")
    with pytest.raises(WorkerStartupError, match="failed during preparation"):
        ExternalWorkerPool(placement, limits, identity, launches, tmp_path / "logs")


def test_corrupt_worker_stream_fails_entire_pool_and_all_children_exit(tmp_path, external_fixture):
    placement, identity, limits, launches = external_fixture
    pool = ExternalWorkerPool(placement, limits, identity, launches, tmp_path / "logs")
    try:
        pool.peers[placement.owners[0]].connection.sendall(PREFIX.pack(b"BADMAGIC", 1, 0) + b"x")
        with pytest.raises((EOFError, OSError)):
            pool.execute(3, ((0.25,) * 32,), ((0, 1, 2, 3),), ((0.0, 0.25, 0.375, 0.375),), ((0.0,) * 32,), 2.5)
        assert pool.closed
        assert all(process.poll() == 0 for process in pool.processes.values())
    finally:
        pool.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_external_bounds_fail_before_creating_worker_files(tmp_path, external_fixture, timeout):
    placement, identity, limits, launches = external_fixture
    with pytest.raises(ValueError, match="timeouts"):
        ExternalWorkerPool(placement, limits, identity, launches, tmp_path / "logs", timeout_seconds=timeout)
    assert not (tmp_path / "logs").exists()


def test_arbitrary_experts_and_compact_token_map_preserve_route_order():
    request = ExpertRequest(
        "id", 3, "npu:0", (Route(4, 2, 6), Route(1, 7, 1), Route(4, 0, 1)), (4, 1), ((1.0,), (2.0,))
    )
    assert compact_routes(request, (1, 6)) == ((0, 1, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="foreign"):
        compact_routes(request, (1, 2))


def test_native_gate_refuses_full_seed_without_initializing_devices():
    placement = glm_placement()
    identity = WireIdentity.for_placement(placement, "a" * 64)
    with pytest.raises(ValueError, match="bounded correctness"):
        NativeGateBackend(
            "ascend_native_gate",
            {"checkpoint_root": "/unused", "device_index": 0},
            "npu:0",
            placement,
            identity,
            PacketLimits(4096, max_routes=136),
        )


def test_external_config_rejects_foreign_placement_and_unknown_backend(external_fixture):
    placement, identity, limits, launches = external_fixture
    config = {
        "owner": placement.owners[0],
        "placement": placement.to_dict(),
        "limits": asdict(limits),
        "identity": asdict(identity),
        "backend": launches[placement.owners[0]].backend,
    }
    changed = json.loads(json.dumps(config))
    changed["identity"]["placement_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="identity"):
        worker_from_config(changed)
    config["backend"] = {"kind": "arbitrary_import"}
    with pytest.raises(ValueError, match="unknown"):
        worker_from_config(config)


def test_external_stuck_worker_is_reported_without_kill():
    pool = ExternalWorkerPool.__new__(ExternalWorkerPool)
    process = Mock(pid=123)
    process.wait.side_effect = subprocess.TimeoutExpired("worker", 0.01)
    pool.peers, pool.processes = {}, {"npu:0": process}
    pool.exit_timeout_seconds, pool.log_dir = 0.01, Path("/tmp/test-log")
    with pytest.raises(RuntimeError, match="PID 123 has not exited"):
        pool.close()
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_split_comparison_checks_precision_shape_and_finiteness():
    assert compare_rows(((1.0, -0.0),), ((1.0, -0.0),))["bitwise_equal"]
    assert not compare_rows(((1.001,),), ((1.0,),))["bitwise_equal"]
    with pytest.raises(AssertionError, match="parity"):
        compare_rows(((1.1,),), ((1.0,),))
    with pytest.raises(ValueError, match="shape"):
        compare_rows(((1.0,),), ((1.0, 2.0),))
    with pytest.raises(ValueError, match="nonfinite"):
        compare_rows(((float("nan"),),), ((1.0,),))


def test_split_fixture_paths_and_checksums(tmp_path):
    (tmp_path / "fixture").write_bytes(b"x")
    with pytest.raises(ValueError, match="checksum"):
        read_checked(tmp_path, "fixture", "0" * 64)
    with pytest.raises(ValueError, match="unsafe"):
        read_checked(tmp_path, "../outside", "0" * 64)
    placement = gate_placement(33)
    assert placement.layer(33)[5] == "cuda:0"
    with pytest.raises(ValueError):
        gate_placement(45)


def test_cuda_combine_packing_preserves_zero_and_rejects_unrepresentable_values():
    assert pack_float_rows(((0.0, -0.0),)) == struct.pack("<ff", 0.0, -0.0)
    for value in (1e100, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="FP32"):
            pack_float_rows(((value,),))
