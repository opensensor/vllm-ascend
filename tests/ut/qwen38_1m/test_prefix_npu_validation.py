# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise diagnostic cleanup and receipts with real CPU checkpoint tiers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.prefix_mamba_state import PrefixMambaStateTier


@pytest.fixture
def diagnostic(monkeypatch):
    parent = ModuleType("tools.qwen4exp.resident_worker")
    parent.QwenResidentExtension = object
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    path = Path(__file__).resolve().parents[3] / "tools/qwen4exp/prefix_npu_validation.py"
    spec = importlib.util.spec_from_file_location("prefix_pressure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tiers = {group: PrefixMambaStateTier([(torch.zeros(64, 2),)], 64, 175) for group in (1, 2, 3)}
    instance = module.QwenPrefixValidationExtension()
    instance.model_runner = SimpleNamespace(_prefix_mamba_tiers=tiers)
    instance.resident_status = lambda: {
        "rank": 2,
        "prefix_mamba": {str(group): tier.cache_status() for group, tier in tiers.items()},
    }
    return instance, tiers


@pytest.mark.parametrize("policy", ["baseline", "bounded"])
def test_pressure_preserves_values_and_cleans_up(diagnostic, policy):
    instance, tiers = diagnostic
    result = instance.resident_prefix_pressure(policy)
    assert result["rank"] == 2
    assert result["value_checks_passed"] and result["storage_preserved"]
    for tier in result["after"]["prefix_mamba"].values():
        if policy == "baseline":
            assert tier["spill_count"] > 0 and tier["restore_count"] == 1
        else:
            assert tier["spill_count"] == tier["restore_count"] == 0
            assert tier["retirement_count"] > 0
    assert all(not (tier._resident or tier._host or tier._device_archive_resident) for tier in tiers.values())


def test_pressure_rejects_live_checkpoints_without_mutation(diagnostic):
    instance, tiers = diagnostic
    tier = tiers[1]
    tier._resident[7] = 1
    before = tier.cache_status()
    with pytest.raises(RuntimeError, match="Drain and reset"):
        instance.resident_prefix_pressure("bounded")
    assert tier.cache_status() == before


def test_pressure_failure_still_clears_all_groups(diagnostic, monkeypatch):
    instance, tiers = diagnostic
    monkeypatch.setattr(tiers[2], "remap_rows", lambda *args: (_ for _ in ()).throw(RuntimeError("probe failed")))
    with pytest.raises(RuntimeError, match="probe failed"):
        instance.resident_prefix_pressure("bounded")
    assert all(not (tier._resident or tier._host or tier._device_archive_resident) for tier in tiers.values())
