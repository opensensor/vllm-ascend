# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.qwen_prefix_replay_scheduler import QwenPrefixReplayScheduler, replay_checkpoint_split


@pytest.fixture
def config(monkeypatch):
    utils = ModuleType("vllm_ascend.utils")
    utils.is_310p = lambda: True
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen4_exp_text")),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_cache_mode="align"),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("alignment, expected", [(128, 128), (None, 64)])
def test_constructor_uses_coordinator_alignment_with_hash_fallback(config, monkeypatch, alignment, expected):
    def parent(self, *args, **kwargs):
        coordinator = SimpleNamespace()
        if alignment is not None:
            coordinator._cache_hit_alignment_tokens = alignment
        self.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
        self.hash_block_size = 64

    monkeypatch.setattr(Scheduler, "__init__", parent)
    assert QwenPrefixReplayScheduler(config)._qwen_replay_alignment == expected


@pytest.mark.parametrize(
    "target, field, value",
    [
        ("model_config.hf_text_config", "model_type", "other_model"),
        ("scheduler_config", "async_scheduling", True),
        ("cache_config", "enable_prefix_caching", False),
        ("cache_config", "mamba_cache_mode", "all"),
        (None, "kv_transfer_config", object()),
    ],
)
def test_incompatible_modes_fail_before_parent_initialization(config, monkeypatch, target, field, value):
    obj = config
    for part in (target or "").split("."):
        if part:
            obj = getattr(obj, part)
    setattr(obj, field, value)
    monkeypatch.setattr(Scheduler, "__init__", lambda *args, **kwargs: pytest.fail("parent must not run"))
    with pytest.raises(ValueError):
        QwenPrefixReplayScheduler(config)


def test_other_hardware_fails_before_parent_initialization(config, monkeypatch):
    monkeypatch.setattr(sys.modules["vllm_ascend.utils"], "is_310p", lambda: False)
    monkeypatch.setattr(Scheduler, "__init__", lambda *args, **kwargs: pytest.fail("parent must not run"))
    with pytest.raises(ValueError, match="Ascend 310P"):
        QwenPrefixReplayScheduler(config)


@pytest.mark.parametrize(
    "start,scheduled,prompt,alignment,expected",
    [
        (7680, 512, 8192, 128, 384),
        (0, 512, 8192, 128, 512),
        (8064, 128, 8192, 128, 128),
        (8192, 3, 8192, 128, 3),
        (7680, 512, 8192, 512, 512),
        (0, 32, 32, 128, 32),
        (0, 0, 8192, 128, 0),
        (1536, 512, 1920, 128, 256),
    ],
)
def test_first_replay_checkpoint_preserves_budget_and_decode(start, scheduled, prompt, alignment, expected):
    assert replay_checkpoint_split(start, scheduled, prompt, alignment) == expected


def test_initial_prefill_materializes_identical_replay_and_extension_endpoints():
    end = 8192
    position, stops = 0, []
    while position < end:
        position += replay_checkpoint_split(position, min(512, end - position), end, 128)
        stops.append(position)
    assert stops[-2:] == [8064, 8192]
    assert all(0 < b - a <= 512 for a, b in zip([0, *stops], stops))


def test_admission_offsets_and_parent_rules_are_preserved(monkeypatch):
    scheduler = QwenPrefixReplayScheduler.__new__(QwenPrefixReplayScheduler)
    scheduler._qwen_replay_alignment = 128
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=8192)
    calls = []

    def parent(self, request, count, local, external):
        calls.append((count, local, external))
        return count - 1

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", parent)
    assert scheduler._mamba_block_aligned_split(request, 512, 4096, 3584) == 383
    assert calls == [(384, 4096, 3584)]


@pytest.mark.parametrize("args", [(-1, 512, 8192, 128), (0, -1, 8192, 128), (0, 512, 0, 128), (0, 512, 8192, 0)])
def test_invalid_replay_geometry_fails(args):
    with pytest.raises(ValueError):
        replay_checkpoint_split(*args)
