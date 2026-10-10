# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned retirement, prefix miss safety and bounded state growth."""

import ast
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend._310p.prefix_mamba_state import (
    PrefixMambaStateTier,
    apply_prefix_mamba_updates,
    retain_prefix_mamba_blocks,
)
from vllm_ascend.core import prefix_mamba_scheduler as mod

pytestmark = pytest.mark.skipif(
    not hasattr(BlockPool, "_insert_block_hash"),
    reason="requires the OpenSensor vLLM prefix-hash alias API (qualified revision 3ab5dda29)",
)


def cache_block(pool, block, group, value):
    key = make_block_hash_with_group_id(value.to_bytes(8, "big"), group)
    pool._insert_block_hash(key, block, value)
    return key


def pool_with_cached_blocks(group=1, count=5):
    pool = BlockPool(32, True, 128, False)
    blocks = pool.get_new_blocks(count)
    keys = [cache_block(pool, block, group, i + 1) for i, block in enumerate(blocks)]
    return pool, blocks, keys


def test_evicted_hash_cannot_produce_a_hit_but_owned_bytes_remain():
    pool, blocks, keys = pool_with_cached_blocks()
    manager = SimpleNamespace(req_to_blocks={"running": [blocks[0], blocks[-1], pool.null_block]})
    result = mod.bounded_prefix_mamba_blocks(pool, {1: manager}, 2)
    assert result == {1: [blocks[0].block_id, blocks[-2].block_id, blocks[-1].block_id]}
    assert pool.get_cached_block(keys[0][:-4], [1]) is None
    assert pool.get_cached_block(keys[-1][:-4], [1]) == [blocks[-1]]
    assert blocks[0].ref_cnt == 1 and blocks[0].block_hash is None
    assert len(pool.cached_block_hash_to_block) == 2


def test_eviction_is_per_mamba_group_and_leaves_attention_hashes():
    pool, blocks, keys = pool_with_cached_blocks(group=0, count=1)
    mamba = pool.get_new_blocks(6)
    for i, block in enumerate(mamba):
        cache_block(pool, block, 1 if i < 3 else 2, 10 + i)
    managers = {group: SimpleNamespace(req_to_blocks={}) for group in (1, 2)}
    result = mod.bounded_prefix_mamba_blocks(pool, managers, 1)
    assert result == {1: [mamba[2].block_id], 2: [mamba[5].block_id]}
    assert pool.get_cached_block(keys[0][:-4], [0]) == blocks
    assert len(pool.cached_block_hash_to_block) == 3


def test_eviction_removes_partial_aliases_before_worker_retirement():
    pool, blocks, keys = pool_with_cached_blocks(count=2)
    alias = cache_block(pool, blocks[0], 1, 99)
    assert pool.get_cached_block(alias[:-4], [1]) == [blocks[0]]
    retained = mod.bounded_prefix_mamba_blocks(pool, {1: SimpleNamespace(req_to_blocks={})}, 1)
    assert retained == {1: [blocks[1].block_id]}
    assert pool.get_cached_block(keys[0][:-4], [1]) is None
    assert pool.get_cached_block(alias[:-4], [1]) is None


def test_recycled_block_with_attention_hash_is_not_retained_as_mamba():
    pool, blocks, _ = pool_with_cached_blocks(count=1)
    manager = SimpleNamespace(req_to_blocks={})
    tracked = {1: {blocks[0].block_id: None}}
    assert mod.bounded_prefix_mamba_blocks(pool, {1: manager}, 1, tracked) == {1: [blocks[0].block_id]}
    pool.evict_blocks({blocks[0].block_id})
    attention_key = cache_block(pool, blocks[0], 0, 9)
    assert mod.bounded_prefix_mamba_blocks(pool, {1: manager}, 1, tracked) == {1: []}
    assert tracked == {1: {}}
    assert pool.get_cached_block(attention_key[:-4], [0]) == blocks


@pytest.mark.parametrize(
    "requests,depth,limit", [(1, 2, 51), (3, 2, 32), (4, 2, 32), (4, 0, 47), (6, 2, 32), (8, 2, 32)]
)
def test_checkpoint_budget_reserves_live_speculative_and_cow_windows(requests, depth, limit):
    assert mod.prefix_mamba_checkpoint_limit(requests, depth) == limit


@pytest.mark.parametrize("requests,depth", [(0, 2), (3, -1)])
def test_unworkable_checkpoint_budget_is_rejected(requests, depth):
    with pytest.raises(ValueError):
        mod.prefix_mamba_checkpoint_limit(requests, depth)


def test_schedule_protects_unowned_cow_source_and_snapshot_survives_worker_ipc(monkeypatch):
    pool, blocks, _ = pool_with_cached_blocks()
    scheduler = object.__new__(mod.PrefixMambaBoundedScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool)
    scheduler._prefix_mamba_managers = {1: SimpleNamespace(req_to_blocks={})}
    scheduler._prefix_mamba_checkpoint_limit = 1
    scheduler._prefix_mamba_tracked = {1: {block.block_id: None for block in blocks}}
    output = SchedulerOutput.make_empty()
    output.kv_cache_block_copies = [(blocks[0].block_id, blocks[1].block_id)]
    parent = Mock(return_value=output)
    monkeypatch.setattr(mod.Scheduler, "schedule", parent)
    result = scheduler.schedule(throttle_prefills=True)
    parent.assert_called_once_with(throttle_prefills=True)
    assert result.ascend_prefix_mamba_block_ids == {
        1: sorted([blocks[0].block_id, blocks[1].block_id, blocks[-1].block_id])
    }
    assert pickle.loads(pickle.dumps(result)).ascend_prefix_mamba_block_ids == result.ascend_prefix_mamba_block_ids


def test_many_retained_histories_stay_on_device_and_latest_prefix_is_exact(monkeypatch):
    pool = BlockPool(1024, True, 128, False)
    states = torch.zeros((64, 2), dtype=torch.float32)
    tier = PrefixMambaStateTier([(states,)], 64)
    snapshot = Mock(side_effect=AssertionError("bounded history must not transfer to CPU"))
    monkeypatch.setattr(tier, "_snapshot", snapshot)
    manager = SimpleNamespace(req_to_blocks={})
    last = None
    last_key = None
    last_value = 0
    oldest_key = None
    tracked = {1: {}}
    # This exceeds primary and archive capacities observed on the live server.
    # A real BlockPool owns hashes/refcounts; only the recurrent math is CPU.
    for step in range(400):
        block = pool.get_new_blocks(1)[0]
        owned = ([last] if last is not None else []) + [block]
        manager.req_to_blocks["request"] = owned
        retained = mod.bounded_prefix_mamba_blocks(pool, {1: manager}, 27, tracked)
        tier.retain_blocks(retained[1])
        table = np.array([[old.block_id for old in owned]], dtype=np.int32)
        mapped = tier.remap_table(table, len(owned))
        if last is not None:
            assert states[mapped[0, 0]].tolist() == [last_value, last_value]
        last_value = step + 1
        states[mapped[0, -1]].fill_(last_value)
        last_key = cache_block(pool, block, 1, step + 1)
        if oldest_key is None:
            oldest_key = last_key
        if last is not None:
            pool.free_blocks([last])
        manager.req_to_blocks["request"] = [block]
        last = block
        assert tier.cache_status()["host_checkpoints"] == 0
        assert len(tier._resident) <= 29
    assert tier.cache_status()["spill_count"] == tier.cache_status()["restore_count"] == 0
    assert tier.cache_status()["retirement_count"] > 300
    assert pool.get_cached_block(oldest_key[:-4], [1]) is None
    hit = pool.get_cached_block(last_key[:-4], [1])
    assert hit == [last]
    pool.touch(hit)
    manager.req_to_blocks["branch"] = hit
    tier.retain_blocks(mod.bounded_prefix_mamba_blocks(pool, {1: manager}, 27)[1])
    mapped = tier.remap_table(np.array([[last.block_id]], dtype=np.int32), 1)
    assert states[mapped[0, 0]].tolist() == [400, 400]
    snapshot.assert_not_called()


def test_three_requests_preserve_distinct_live_states_without_spill_or_pool_scan(monkeypatch):
    pool = BlockPool(2048, True, 128, False)

    class IndexedBlocks:
        def __len__(self):
            return len(blocks)

        def __getitem__(self, index):
            return blocks[index]

        def __iter__(self):
            raise AssertionError("per-step decode scanned every attention KV block")

    blocks = pool.blocks
    pool.blocks = IndexedBlocks()
    states = torch.zeros((64, 2), dtype=torch.float32)
    tier = PrefixMambaStateTier([(states,)], 64)
    monkeypatch.setattr(tier, "_snapshot", Mock(side_effect=AssertionError("unexpected spill")))
    manager = SimpleNamespace(req_to_blocks={})
    tracked = {1: {}}
    previous = {}
    for step in range(400):
        current = dict(zip(range(3), pool.get_new_blocks(3), strict=True))
        manager.req_to_blocks = {str(row): ([previous[row]] if previous else []) + [current[row]] for row in range(3)}
        tier.retain_blocks(mod.bounded_prefix_mamba_blocks(pool, {1: manager}, 27, tracked)[1])
        table = np.array([[b.block_id for b in owned] for owned in manager.req_to_blocks.values()], dtype=np.int32)
        mapped = tier.remap_rows(table, [table.shape[1]] * 3, [tuple(range(table.shape[1]))] * 3)
        for row in range(3):
            if previous:
                assert states[mapped[row, 0]].tolist() == [row * 1000 + step] * 2
            states[mapped[row, -1]].fill_(row * 1000 + step + 1)
            cache_block(pool, current[row], 1, step * 3 + row + 1)
        if previous:
            pool.free_blocks(list(previous.values()))
        previous = current
    assert tier.cache_status()["spill_count"] == tier.cache_status()["restore_count"] == 0
    assert tier.cache_status()["retirement_count"] > 1000


def runner_class():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/model_runner_310p.py"
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310")
    update = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "_update_states")

    class Base:
        def _update_states(self, output):
            self.events.append("base_update")
            return "updated"

    namespace = {
        "Base": Base,
        "SchedulerOutput": object,
        "MultiGroupBlockTable310": object,
        "cast": lambda cls, value: value,
        "retain_prefix_mamba_blocks": retain_prefix_mamba_blocks,
        "apply_prefix_mamba_updates": apply_prefix_mamba_updates,
        "torch": SimpleNamespace(npu=SimpleNamespace(synchronize=Mock())),
    }
    extracted = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[update], decorator_list=[]
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), str(path), "exec"), namespace
    )
    return namespace["Runner"]


@pytest.mark.parametrize("snapshot", [None, {1: [101, 102]}])
def test_runner_consumes_authoritative_snapshot_before_recycled_ids(snapshot):
    runner = runner_class()()
    runner.events = []
    tier = SimpleNamespace(
        _resident={},
        _device_archive_resident={},
        retain_blocks=Mock(side_effect=lambda ids, **kwargs: runner.events.append(("retain", ids))),
        invalidate=Mock(side_effect=lambda ids, **kwargs: runner.events.append(("invalidate", ids))),
    )
    runner._prefix_mamba_tiers = {1: tier}
    runner._new_prefix_mamba_block_ids = lambda _: {1: {103}}
    runner.input_batch = SimpleNamespace(block_table=object())
    runner.mamba_state_idx = {}
    output = SchedulerOutput.make_empty()
    if snapshot is not None:
        output.ascend_prefix_mamba_block_ids = snapshot
    assert runner._update_states(output) == "updated"
    assert runner.events == ([("retain", [101, 102])] if snapshot else []) + ["base_update", ("invalidate", {103})]


def test_runner_rejects_incomplete_groups_before_retiring_any_state():
    runner = runner_class()()
    tier = SimpleNamespace(retain_blocks=Mock())
    runner._prefix_mamba_tiers = {1: tier, 2: tier}
    runner._new_prefix_mamba_block_ids = lambda _: {}
    output = SchedulerOutput.make_empty()
    output.ascend_prefix_mamba_block_ids = {1: [101]}
    with pytest.raises(RuntimeError, match="does not match worker cache groups"):
        runner._update_states(output)
    tier.retain_blocks.assert_not_called()


def valid_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen4_exp_text")),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=3),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_cache_mode="align"),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
    )


@pytest.mark.parametrize("depth,expected", [(None, 51), (2, 32)])
def test_scheduler_constructor_preserves_parent_setup_and_tracks_mamba_groups(monkeypatch, depth, expected):
    config = valid_config()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=depth) if depth is not None else None
    monkeypatch.setattr(mod, "_is_310p", lambda: True)
    manager = object.__new__(mod.MambaManager)
    manager.kv_cache_group_id = 1

    def parent(self, cfg, *args, **kwargs):
        assert cfg is config and args == ("kv-config",) and kwargs == {"log_stats": True}
        self.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[object(), manager]))

    monkeypatch.setattr(mod.Scheduler, "__init__", parent)
    scheduler = mod.PrefixMambaBoundedScheduler(config, "kv-config", log_stats=True)
    assert scheduler._prefix_mamba_checkpoint_limit == expected
    assert scheduler._prefix_mamba_managers == {1: manager}
    assert scheduler._prefix_mamba_tracked == {1: {}}


def test_scheduler_constructor_rejects_a_runner_without_mamba_groups(monkeypatch):
    monkeypatch.setattr(mod, "_is_310p", lambda: True)
    monkeypatch.setattr(
        mod.Scheduler,
        "__init__",
        lambda self, *args: setattr(
            self, "kv_cache_manager", SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[]))
        ),
    )
    with pytest.raises(ValueError, match="no Mamba cache groups"):
        mod.PrefixMambaBoundedScheduler(valid_config())


@pytest.mark.parametrize("unsupported", ["hardware", "model", "async", "transfer", "prefix", "mode"])
def test_scheduler_rejects_unsupported_serving_before_initializing(monkeypatch, unsupported):
    config = valid_config()
    monkeypatch.setattr(mod, "_is_310p", lambda: unsupported != "hardware")
    if unsupported == "model":
        config.model_config.hf_text_config.model_type = "other"
    elif unsupported == "async":
        config.scheduler_config.async_scheduling = True
    elif unsupported == "transfer":
        config.kv_transfer_config = object()
    elif unsupported == "prefix":
        config.cache_config.enable_prefix_caching = False
    elif unsupported == "mode":
        config.cache_config.mamba_cache_mode = "none"
    parent = Mock()
    monkeypatch.setattr(mod.Scheduler, "__init__", parent)
    with pytest.raises(ValueError):
        mod.PrefixMambaBoundedScheduler(config)
    parent.assert_not_called()
