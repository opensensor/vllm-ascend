# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU placement-simulation UT for the Qwen4Exp 1M streamed sharded loader (T3.2).

Drives the placement simulator from the in-repo checkpoint manifest (metadata
only) and validates that a TP4/EP4 load:

* never materialises the full 512-expert bank on any rank,
* lands within the measured 31.88 GiB/chip non-PLE target (+ manifest drift),
* keeps the 95.43 GiB PLE table host-resident (excluded from device totals),
* places every tensor exactly once (no double instantiation), and
* walks manifest metadata with a bounded host working set (never 224 GB).

Run: ``python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_shard_placement.py``
"""

import json
import tracemalloc
from pathlib import Path

import pytest

from vllm_ascend._310p.sharded_state_loader_310p import (
    PLACEMENT_DRIFT_TOLERANCE,
    TARGET_NON_PLE_PER_CHIP_BYTES,
    TARGET_NON_PLE_PER_CHIP_GIB,
    DoubleInstantiationError,
    Qwen4ExpPlacementSimulator,
    ShardPolicy,
    read_index_component_bytes,
)
from vllm_ascend.observability.qwen38_mem_accounting import (
    MAX_PLACEMENT_IMBALANCE,
    MemComponent,
)

_GIB = 1024**3
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPO_ROOT / "artifacts" / "qwen38-1m" / "checkpoint-manifest.json"

# Reconciliation tolerance between the analytic simulation and the checkpoint's
# reported byte totals. The only gap is a ~0.48 MB PLE rounding term and shard
# header packaging (< 0.02 %); 0.5 % leaves generous headroom.
_BYTE_RECONCILE_TOL = 0.005


@pytest.fixture(scope="module")
def manifest() -> dict:
    with _MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def tp4_report(manifest: dict) -> dict:
    return Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp").predicted_report()


# --------------------------------------------------------------------------- #
# Completeness: every manifest tensor is enumerated exactly once.
# --------------------------------------------------------------------------- #
def test_tensor_count_reconciles_to_manifest(tp4_report: dict, manifest: dict):
    assert tp4_report["tensor_total"] == manifest["tensor_total"]


def test_no_tensor_instantiated_twice(manifest: dict):
    sim = Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp")
    sim.simulate()  # raises DoubleInstantiationError on any double-place/drop
    ledger = sim.ledger
    # Every sharded byte accounted for exactly once across the four ranks.
    assert ledger.sharded_placed_bytes == ledger.sharded_total_bytes
    assert ledger.sharded_total_bytes > 100 * _GIB
    # The routed-expert weight bank (the thing that must never be duplicated) is
    # the dominant sharded contributor.
    assert ledger.device_logical_bytes > 120 * _GIB


def test_double_instantiation_is_detected():
    from vllm_ascend._310p.sharded_state_loader_310p import PlacementLedger

    bad = PlacementLedger(sharded_total_bytes=100, sharded_placed_bytes=200)
    with pytest.raises(DoubleInstantiationError):
        bad.assert_no_double_instantiation()


# --------------------------------------------------------------------------- #
# Byte reconciliation against the real checkpoint totals (from the manifest).
# --------------------------------------------------------------------------- #
def test_total_bytes_reconcile_with_manifest(manifest: dict):
    sim = Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp")
    sim.simulate()
    ledger = sim.ledger
    total_logical = ledger.device_logical_bytes + ledger.host_logical_bytes

    export_bytes = manifest["conversion_report"]["export_tensor_bytes"]
    shard_bytes = sum(s["bytes"] for s in manifest["shards"])

    assert abs(total_logical - export_bytes) / export_bytes < _BYTE_RECONCILE_TOL
    assert abs(total_logical - shard_bytes) / shard_bytes < _BYTE_RECONCILE_TOL


# --------------------------------------------------------------------------- #
# Per-chip device footprint vs the measured 31.88 GiB/chip target.
# --------------------------------------------------------------------------- #
def test_per_chip_device_within_target_plus_drift(tp4_report: dict):
    max_bytes = tp4_report["max_per_chip_device_bytes"]
    ceiling = TARGET_NON_PLE_PER_CHIP_BYTES * (1 + PLACEMENT_DRIFT_TOLERANCE)
    assert max_bytes <= ceiling, (
        f"per-chip {max_bytes / _GIB:.3f} GiB exceeds "
        f"{TARGET_NON_PLE_PER_CHIP_GIB} GiB +{PLACEMENT_DRIFT_TOLERANCE:.0%}"
    )
    # And meaningfully near target (not trivially small): within -3%..+3%.
    assert max_bytes >= TARGET_NON_PLE_PER_CHIP_BYTES * (1 - PLACEMENT_DRIFT_TOLERANCE)
    # Documented drift stays under ~1 %.
    assert tp4_report["drift_fraction"] < 0.01


def test_per_chip_balanced(tp4_report: dict):
    per_chip = tp4_report["per_chip_device_bytes"]
    assert set(per_chip) == {0, 1, 2, 3}
    mean = sum(per_chip.values()) / len(per_chip)
    imbalance = max(abs(v - mean) / mean for v in per_chip.values())
    assert imbalance <= MAX_PLACEMENT_IMBALANCE


def test_no_rank_holds_full_expert_bank(manifest: dict):
    sim = Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp")
    acc = sim.simulate()
    expert_per_rank = [r.components.get(MemComponent.EXPERT_W8A8, 0) for r in acc.ranks.values()]
    total_expert = sum(expert_per_rank)
    assert total_expert > 110 * _GIB
    for rank_bytes in expert_per_rank:
        # Each rank holds ~1/4 of the bank, never the whole thing.
        assert rank_bytes == pytest.approx(total_expert / 4, rel=0.001)
        assert rank_bytes < total_expert


# --------------------------------------------------------------------------- #
# PLE stays host-resident and is counted once.
# --------------------------------------------------------------------------- #
def test_ple_host_resident_and_counted_once(tp4_report: dict):
    acc = tp4_report["accountant"]
    host_bytes = tp4_report["host_table_bytes"]
    # ~95.43 GiB, matching the runtime-requirements doc.
    assert host_bytes / _GIB == pytest.approx(95.43, abs=0.1)
    ple_key = MemComponent.PLE_HOST_TABLE.value
    for rank in acc["ranks"]:
        # The PLE table is recorded on the rank (host-resident) but is a host
        # component: it is excluded from the device total.
        assert ple_key in rank["components"]
        assert rank["components"][ple_key] == host_bytes
        assert rank["device_bytes"] < 40 * _GIB
        # device_bytes excludes the ~95 GiB host table.
        assert rank["device_bytes"] < host_bytes
    # Host table is a single shared logical copy, not multiplied by world_size.
    assert host_bytes < 100 * _GIB


# --------------------------------------------------------------------------- #
# Bounded host working set: the walk never instantiates 224 GB.
# --------------------------------------------------------------------------- #
def test_simulation_working_set_is_bounded(manifest: dict):
    sim = Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp")
    tracemalloc.start()
    tracemalloc.reset_peak()
    sim.simulate()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Accounted the whole model...
    accounted = sim.ledger.device_logical_bytes + sim.ledger.host_logical_bytes
    assert accounted > 200 * _GIB
    # ...while holding at most a few MB of host RAM (never the 224 GB payload).
    assert peak < 64 * 1024 * 1024, f"peak host RSS {peak / 1e6:.1f} MB too large"
    # Only one tensor descriptor is live at a time.
    assert sim.ledger.peak_working_set_objects == 1


# --------------------------------------------------------------------------- #
# EP4 path readiness: expert-parallel placement is balanced and dup-free.
# --------------------------------------------------------------------------- #
def test_ep4_matches_tp4_device_footprint(manifest: dict, tp4_report: dict):
    ep = Qwen4ExpPlacementSimulator(manifest, ep_size=4, parallel_mode="ep")
    ep_report = ep.predicted_report()
    # Same aggregate device bytes, same balanced per-chip footprint.
    assert ep_report["device_aggregate_bytes"] == tp4_report["device_aggregate_bytes"]
    ep_max = ep_report["max_per_chip_device_bytes"]
    tp_max = tp4_report["max_per_chip_device_bytes"]
    assert ep_max == pytest.approx(tp_max, rel=1e-6)
    # No double instantiation on the EP path either.
    ep.simulate()
    assert ep.ledger.sharded_placed_bytes == ep.ledger.sharded_total_bytes


def test_ep4_expert_bank_partitioned_across_ranks(manifest: dict):
    ep = Qwen4ExpPlacementSimulator(manifest, ep_size=4, parallel_mode="ep")
    acc = ep.simulate()
    expert_per_rank = [r.components.get(MemComponent.EXPERT_W8A8, 0) for r in acc.ranks.values()]
    total_expert = sum(expert_per_rank)
    for rank_bytes in expert_per_rank:
        assert rank_bytes == pytest.approx(total_expert / 4, rel=0.001)


def test_norm_family_is_replicated(manifest: dict):
    # RMS/layer norms are the only replicated family; verify the policy wiring.
    sim = Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp")
    norm_groups = [g for g in sim.iter_dense_tensors() if g.name == "norm"]
    assert len(norm_groups) == 1
    assert norm_groups[0].policy is ShardPolicy.REPLICATED


# --------------------------------------------------------------------------- #
# Optional authoritative cross-check against the real safetensors headers.
# --------------------------------------------------------------------------- #
def test_optional_index_crosscheck(manifest: dict):
    totals = read_index_component_bytes(manifest["checkpoint_dir"])
    if totals is None:
        pytest.skip("checkpoint volume not mounted; manifest-driven sim already validated")
    host = {"ple_ngram", "ple_other"}
    device_header_bytes = sum(b for c, b in totals.items() if c not in host)
    host_header_bytes = sum(b for c, b in totals.items() if c in host)

    sim = Qwen4ExpPlacementSimulator(manifest, tp_size=4, parallel_mode="tp")
    sim.simulate()
    assert abs(sim.ledger.device_logical_bytes - device_header_bytes) / device_header_bytes < _BYTE_RECONCILE_TOL
    assert abs(sim.ledger.host_logical_bytes - host_header_bytes) / host_header_bytes < _BYTE_RECONCILE_TOL
