# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek V4.1 W2 per-rank memory accountant.

Runs host-side only (pure Python, synthetic allocation traces). Execute with
``--noconftest`` because the shared ``tests/ut/conftest.py`` is broken:

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_mem_accounting.py
"""

from __future__ import annotations

import pytest

from vllm_ascend.observability.deepseek_w2_mem_accounting import (
    DEEPSEEK_HOST_COMPONENTS,
    HOST_COMPONENTS,
    MAX_PLACEMENT_IMBALANCE,
    DeepSeekW2MemComponent,
    DeepSeekW2MemoryAccountant,
    DeepSeekW2RankMemoryReport,
    PlacementImbalanceError,
)
from vllm_ascend.observability.qwen38_mem_accounting import (
    HOST_COMPONENTS as QWEN_HOST_COMPONENTS,
)
from vllm_ascend.observability.qwen38_mem_accounting import (
    MemComponent,
)

# --- Synthetic trace constants (raw bytes; values chosen for clean sums) ------

W2_EXPERT_BYTES = 12_000_000_000  # packed 2-bit experts (device / HBM), per rank
UNPACK_CACHE_BYTES = 1_500_000_000  # INT8 active-expert unpack cache (device), per rank
NON_EXPERT_BYTES = 800_000_000  # reused Qwen device component, per rank
ENGRAM_HOST_BYTES = 250_000_000_000  # ~W4 Engram host table, single shared copy


def _balanced_accountant(world_size: int = 4) -> DeepSeekW2MemoryAccountant:
    acc = DeepSeekW2MemoryAccountant(world_size=world_size)
    for rank in range(world_size):
        rep = acc.rank_report(rank)
        rep.add(DeepSeekW2MemComponent.W2_EXPERT, W2_EXPERT_BYTES)
        rep.add(DeepSeekW2MemComponent.UNPACK_CACHE, UNPACK_CACHE_BYTES)
        rep.add(MemComponent.NON_EXPERT_FP16, NON_EXPERT_BYTES)
        # Engram host table: identical single shared copy recorded on every rank.
        rep.add(DeepSeekW2MemComponent.ENGRAM_HOST, ENGRAM_HOST_BYTES)
    return acc


# --- Component classification -------------------------------------------------


def test_engram_is_host_component():
    assert DeepSeekW2MemComponent.ENGRAM_HOST in DEEPSEEK_HOST_COMPONENTS
    assert DeepSeekW2MemComponent.ENGRAM_HOST in HOST_COMPONENTS


def test_w2_and_unpack_cache_are_device_components():
    assert DeepSeekW2MemComponent.W2_EXPERT not in HOST_COMPONENTS
    assert DeepSeekW2MemComponent.UNPACK_CACHE not in HOST_COMPONENTS


def test_combined_host_set_unions_qwen_ple_table():
    # The DeepSeek report must still classify the Qwen PLE host table as host.
    assert QWEN_HOST_COMPONENTS <= HOST_COMPONENTS
    assert MemComponent.PLE_HOST_TABLE in HOST_COMPONENTS


# --- Component sums match the synthetic trace ---------------------------------


def test_device_bytes_sum_excludes_engram_host():
    acc = _balanced_accountant()
    expected_device = W2_EXPERT_BYTES + UNPACK_CACHE_BYTES + NON_EXPERT_BYTES
    for rank in range(4):
        assert acc.rank_report(rank).device_bytes() == expected_device
        # Engram bytes live on the host side, not device.
        assert acc.rank_report(rank).host_bytes() == ENGRAM_HOST_BYTES


def test_report_is_deepseek_report_type():
    acc = _balanced_accountant()
    assert isinstance(acc.rank_report(0), DeepSeekW2RankMemoryReport)


def test_add_accumulates():
    rep = DeepSeekW2RankMemoryReport(rank=0)
    rep.add(DeepSeekW2MemComponent.W2_EXPERT, 100)
    rep.add(DeepSeekW2MemComponent.W2_EXPERT, 25)
    assert rep.components[DeepSeekW2MemComponent.W2_EXPERT] == 125
    assert rep.device_bytes() == 125


def test_add_rejects_negative():
    rep = DeepSeekW2RankMemoryReport(rank=0)
    with pytest.raises(ValueError):
        rep.add(DeepSeekW2MemComponent.W2_EXPERT, -1)


# --- Engram host: counted once, never multiplied by world_size ----------------


def test_engram_host_counted_once_not_times_world_size():
    acc = _balanced_accountant(world_size=4)
    # Single shared logical copy: exactly the per-copy value, NOT ×4.
    assert acc.host_table_bytes() == ENGRAM_HOST_BYTES
    assert acc.host_table_bytes() != ENGRAM_HOST_BYTES * acc.world_size


def test_engram_host_divergence_across_ranks_raises():
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    acc.rank_report(0).add(DeepSeekW2MemComponent.ENGRAM_HOST, ENGRAM_HOST_BYTES)
    acc.rank_report(1).add(DeepSeekW2MemComponent.ENGRAM_HOST, ENGRAM_HOST_BYTES + 1)
    with pytest.raises(ValueError, match="single shared logical copy"):
        acc.host_table_bytes()


def test_host_table_bytes_zero_when_absent():
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    acc.rank_report(0).add(DeepSeekW2MemComponent.W2_EXPERT, W2_EXPERT_BYTES)
    acc.rank_report(1).add(DeepSeekW2MemComponent.W2_EXPERT, W2_EXPERT_BYTES)
    assert acc.host_table_bytes() == 0


# --- Device imbalance guard (5%) ----------------------------------------------


def test_balanced_placement_passes():
    acc = _balanced_accountant()
    assert acc.imbalance() == 0.0
    acc.validate_balance()  # must not raise


def test_imbalance_within_5pct_passes():
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    base = 1_000_000_000
    acc.rank_report(0).add(DeepSeekW2MemComponent.W2_EXPERT, base)
    # +4% on one rank -> max deviation from mean is ~1.96% (< 5%).
    acc.rank_report(1).add(DeepSeekW2MemComponent.W2_EXPERT, int(base * 1.04))
    assert acc.imbalance() <= MAX_PLACEMENT_IMBALANCE
    acc.validate_balance()  # must not raise


def test_imbalance_over_5pct_raises():
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    base = 1_000_000_000
    acc.rank_report(0).add(DeepSeekW2MemComponent.W2_EXPERT, base)
    acc.rank_report(1).add(DeepSeekW2MemComponent.W2_EXPERT, base * 2)
    assert acc.imbalance() > MAX_PLACEMENT_IMBALANCE
    with pytest.raises(PlacementImbalanceError):
        acc.validate_balance()


def test_imbalance_over_5pct_passes_with_approved_reason():
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    base = 1_000_000_000
    acc.rank_report(0).add(DeepSeekW2MemComponent.W2_EXPERT, base)
    acc.rank_report(1).add(DeepSeekW2MemComponent.W2_EXPERT, base * 2)
    # Approved override must not raise.
    acc.validate_balance(approved_reason="asymmetric expert shard for E0.5 bring-up")


def test_engram_host_divergence_does_not_break_device_balance():
    # Host divergence must NOT contaminate the device imbalance check: device
    # bytes are balanced even though Engram bytes differ.
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    for rank in range(2):
        acc.rank_report(rank).add(DeepSeekW2MemComponent.W2_EXPERT, W2_EXPERT_BYTES)
    acc.rank_report(0).add(DeepSeekW2MemComponent.ENGRAM_HOST, ENGRAM_HOST_BYTES)
    acc.rank_report(1).add(DeepSeekW2MemComponent.ENGRAM_HOST, ENGRAM_HOST_BYTES * 3)
    assert acc.imbalance() == 0.0
    acc.validate_balance()  # device is balanced


# --- Serialization / summary --------------------------------------------------


def test_to_dict_reports_single_copy_host_and_device_sums():
    acc = _balanced_accountant()
    d = acc.to_dict()
    assert d["world_size"] == 4
    assert d["host_table_bytes"] == ENGRAM_HOST_BYTES
    assert d["max_imbalance_threshold"] == MAX_PLACEMENT_IMBALANCE
    expected_device = W2_EXPERT_BYTES + UNPACK_CACHE_BYTES + NON_EXPERT_BYTES
    for rank_dict in d["ranks"]:
        assert rank_dict["device_bytes"] == expected_device
        assert rank_dict["host_bytes"] == ENGRAM_HOST_BYTES
        # Every component (device + host) is itemized in the breakdown.
        assert rank_dict["components"]["w2_expert"] == W2_EXPERT_BYTES
        assert rank_dict["components"]["unpack_cache"] == UNPACK_CACHE_BYTES
        assert rank_dict["components"]["engram_host"] == ENGRAM_HOST_BYTES


def test_to_json_roundtrips():
    import json

    acc = _balanced_accountant()
    parsed = json.loads(acc.to_json())
    assert parsed["host_table_bytes"] == ENGRAM_HOST_BYTES


def test_human_summary_mentions_engram_and_omits_host_from_device_lines():
    acc = _balanced_accountant()
    summary = acc.human_summary()
    assert "DeepSeek" in summary
    assert "Engram" in summary
    assert "w2_expert" in summary
    assert "unpack_cache" in summary
    # Engram is host-resident and must not appear as a per-rank device line item.
    assert "engram_host" not in summary


def test_device_totals_are_reported_not_summed():
    acc = _balanced_accountant()
    rep = acc.rank_report(0)
    rep.set_device_totals(free_bytes=7_000_000_000, peak_bytes=9_000_000_000)
    d = rep.to_dict()
    assert d["free_bytes"] == 7_000_000_000
    assert d["peak_bytes"] == 9_000_000_000
    # Free/peak are informational and do not enter the component device sum.
    expected_device = W2_EXPERT_BYTES + UNPACK_CACHE_BYTES + NON_EXPERT_BYTES
    assert d["device_bytes"] == expected_device


def test_rank_out_of_range_raises():
    acc = DeepSeekW2MemoryAccountant(world_size=2)
    with pytest.raises(ValueError):
        acc.rank_report(2)
    with pytest.raises(ValueError):
        acc.rank_report(-1)
