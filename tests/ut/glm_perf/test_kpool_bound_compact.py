# SPDX-License-Identifier: Apache-2.0
"""Prepared converter descriptors survive combined capture without lazy fills."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_candidates import kpool_bound_compact as candidate


class CpuPreparation:
    npu = SimpleNamespace(current_device=lambda: 0)

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, kind, index):
        assert kind == "npu" and index == 0
        return torch.device("cpu")


def test_shared_target_and_draft_converter_rebases_all_existing_descriptors(monkeypatch):
    monkeypatch.setattr(candidate, "torch", CpuPreparation())
    old = torch.tensor([7, 4], dtype=torch.int64)
    key = (torch.device("cpu"), 7, 4)
    converter = SimpleNamespace(configs={key: old})
    modules = [
        SimpleNamespace(_native_bf16_cast=converter, indexer_op=object(), rope_dim=64, head_dim=128, n_head=32)
        for _ in range(2)
    ]
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: modules[:1]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: modules[1:])),
    )
    assert candidate.prepare_converters(runner) == 1
    assert converter.configs[key] is not old and converter.configs[key].tolist() == [7, 4]
    for rows in (1, 2, 8, 160, 640):
        for width in (64, 128, 512, 4096):
            for mode in range(6):
                assert converter.configs[torch.device("cpu"), rows * width, mode].tolist() == [rows * width, mode]
    assert len({value.untyped_storage().data_ptr() for value in converter.configs.values()}) == 1


def test_wrong_device_descriptors_are_rejected_before_rebasing(monkeypatch):
    monkeypatch.setattr(candidate, "torch", CpuPreparation())
    original = {(torch.device("meta"), 256, 5): object()}
    converter = SimpleNamespace(configs=original)
    module = SimpleNamespace(_native_bf16_cast=converter, indexer_op=object(), rope_dim=64, head_dim=128, n_head=32)
    runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: [module]))
    with pytest.raises(ValueError, match="another device"):
        candidate.prepare_converters(runner)
    assert converter.configs is original


def test_missing_model_is_rejected_without_any_device_initialization():
    runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: []))
    with pytest.raises(ValueError, match="no permanent"):
        candidate.prepare_converters(runner)
