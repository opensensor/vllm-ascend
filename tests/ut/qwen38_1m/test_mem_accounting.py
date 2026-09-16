# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Qwen4Exp 310P per-rank memory accounting harness (plan T0.5)."""

import json

import pytest

from vllm_ascend.observability.qwen38_mem_accounting import (
    HOST_COMPONENTS,
    MAX_PLACEMENT_IMBALANCE,
    MemComponent,
    MemoryAccountant,
    PlacementImbalanceError,
    RankMemoryReport,
)

_GIB = 1024**3

# Measured per-chip device footprint from the runtime-requirements doc:
# 31.88 GiB non-PLE weights per chip, split across a couple of components.
_EXPERT_W8A8_PER_CHIP = int(29.30 * _GIB)
_NON_EXPERT_PER_CHIP = int(2.43 * _GIB)
_SCALES_PER_CHIP = int(0.05 * _GIB)
_PLE_HOST_BYTES = int(95.43 * _GIB)


def _balanced_accountant(world_size: int = 4) -> MemoryAccountant:
    accountant = MemoryAccountant(world_size=world_size)
    for rank in range(world_size):
        report = accountant.rank_report(rank)
        report.add(MemComponent.EXPERT_W8A8, _EXPERT_W8A8_PER_CHIP)
        report.add(MemComponent.NON_EXPERT_FP16, _NON_EXPERT_PER_CHIP)
        report.add(MemComponent.QUANT_SCALES, _SCALES_PER_CHIP)
        # PLE table is one shared logical copy; each rank observes the same value.
        report.add(MemComponent.PLE_HOST_TABLE, _PLE_HOST_BYTES)
    return accountant


def test_component_sums_match_synthetic_trace():
    accountant = _balanced_accountant()
    per_chip_device = _EXPERT_W8A8_PER_CHIP + _NON_EXPERT_PER_CHIP + _SCALES_PER_CHIP
    for rank in range(4):
        report = accountant.ranks[rank]
        assert report.device_bytes() == per_chip_device
        # Host component excluded from device total.
        assert report.host_bytes() == _PLE_HOST_BYTES
    # Aggregate device footprint reproduces the ~127.5 GiB measured figure.
    total_device = sum(r.device_bytes() for r in accountant.ranks.values())
    assert total_device / _GIB == pytest.approx(127.5, abs=0.5)


def test_host_table_counted_once_not_per_rank():
    accountant = _balanced_accountant()
    # The 95.43 GiB PLE table must not be multiplied by world_size.
    assert accountant.host_table_bytes() == _PLE_HOST_BYTES


def test_host_table_divergence_across_ranks_raises():
    accountant = MemoryAccountant(world_size=2)
    accountant.rank_report(0).add(MemComponent.PLE_HOST_TABLE, _PLE_HOST_BYTES)
    accountant.rank_report(1).add(MemComponent.PLE_HOST_TABLE, _PLE_HOST_BYTES + _GIB)
    with pytest.raises(ValueError, match="single shared logical copy"):
        accountant.host_table_bytes()


def test_balanced_placement_passes_validation():
    accountant = _balanced_accountant()
    assert accountant.imbalance() == pytest.approx(0.0, abs=1e-9)
    accountant.validate_balance()  # must not raise


def test_imbalance_violation_raises():
    accountant = _balanced_accountant()
    # Overload rank 0 by 10% of its device bytes -> exceeds the 5% limit.
    extra = int(accountant.ranks[0].device_bytes() * 0.10)
    accountant.rank_report(0).add(MemComponent.GDN_STATE, extra)
    assert accountant.imbalance() > MAX_PLACEMENT_IMBALANCE
    with pytest.raises(PlacementImbalanceError, match="imbalance"):
        accountant.validate_balance()


def test_imbalance_override_with_reason():
    accountant = _balanced_accountant()
    extra = int(accountant.ranks[0].device_bytes() * 0.10)
    accountant.rank_report(0).add(MemComponent.GDN_STATE, extra)
    # An approved reason permits the over-threshold placement.
    accountant.validate_balance(approved_reason="rank0 hosts indexer history by design")


def test_imbalance_just_under_threshold_passes():
    accountant = _balanced_accountant()
    # Add 4% to one rank: below the 5% ceiling (max deviation from mean < 5%).
    accountant.rank_report(0).add(MemComponent.GDN_STATE, int(accountant.ranks[0].device_bytes() * 0.04))
    assert accountant.imbalance() <= MAX_PLACEMENT_IMBALANCE
    accountant.validate_balance()


def test_single_rank_reports_zero_imbalance():
    accountant = MemoryAccountant(world_size=1)
    accountant.rank_report(0).add(MemComponent.EXPERT_W8A8, _EXPERT_W8A8_PER_CHIP)
    assert accountant.imbalance() == 0.0
    accountant.validate_balance()


def test_device_totals_and_json_roundtrip():
    accountant = _balanced_accountant()
    accountant.ranks[0].set_device_totals(free_bytes=12 * _GIB, peak_bytes=33 * _GIB)
    payload = json.loads(accountant.to_json())
    assert payload["world_size"] == 4
    assert payload["host_table_bytes"] == _PLE_HOST_BYTES
    assert len(payload["ranks"]) == 4
    rank0 = next(r for r in payload["ranks"] if r["rank"] == 0)
    assert rank0["free_bytes"] == 12 * _GIB
    assert rank0["peak_bytes"] == 33 * _GIB
    assert rank0["components"][MemComponent.EXPERT_W8A8.value] == _EXPERT_W8A8_PER_CHIP
    # Host component appears in the per-rank component map but not device_bytes.
    assert rank0["components"][MemComponent.PLE_HOST_TABLE.value] == _PLE_HOST_BYTES
    assert rank0["device_bytes"] == _EXPERT_W8A8_PER_CHIP + _NON_EXPERT_PER_CHIP + _SCALES_PER_CHIP


def test_human_summary_mentions_ranks_and_host_table():
    accountant = _balanced_accountant()
    summary = accountant.human_summary()
    assert "rank 0" in summary
    assert "host PLE table (shared)" in summary
    assert "95.43 GiB" in summary


def test_negative_bytes_rejected():
    report = RankMemoryReport(rank=0)
    with pytest.raises(ValueError, match=">= 0"):
        report.add(MemComponent.WORKSPACES, -1)


def test_rank_out_of_range_rejected():
    accountant = MemoryAccountant(world_size=4)
    with pytest.raises(ValueError, match="out of range"):
        accountant.rank_report(4)


def test_ple_is_the_only_host_component():
    # Guards the capacity model: only the PLE table is host-resident.
    assert frozenset({MemComponent.PLE_HOST_TABLE}) == HOST_COMPONENTS
