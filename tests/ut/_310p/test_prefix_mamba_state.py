# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend._310p.prefix_mamba_state import (
    PrefixMambaStateTier,
    get_mamba_postprocess_block_ids,
    prefix_mamba_active_columns,
    prefix_mamba_device_archive_slots,
    prefix_mamba_slot_count,
    prefix_mamba_state_bytes_per_slot,
)


def _tier(num_slots: int = 3, archive_slots: int = 0) -> tuple[PrefixMambaStateTier, torch.Tensor]:
    states = torch.zeros((num_slots, 2), dtype=torch.float16)
    return PrefixMambaStateTier([(states,)], num_slots, archive_slots), states


@pytest.mark.parametrize("archive_slots", [0, 2])
def test_reset_discards_all_checkpoint_tiers_without_reallocating(archive_slots, monkeypatch):
    tier, states = _tier(archive_slots=archive_slots)
    for block_id in range(101, 109):
        tier.remap_table(np.array([[block_id]], dtype=np.int32), 1)
        states[tier.slot_for(block_id)].fill_(block_id)
    assert tier._resident and tier._host
    assert bool(tier._device_archive_resident) == bool(archive_slots)
    storage = (states, *tier._device_archive, *tier._swap_tensors)
    pointers = [tensor.data_ptr() for tensor in storage]
    spills = tier._spill_count
    sync = Mock()
    monkeypatch.setattr(tier, "_synchronize_device_state", sync)

    tier.reset()

    sync.assert_called_once_with()
    assert not tier._resident and not tier._device_archive_resident and not tier._host
    assert tier._unused_slots == [2, 1]
    assert tier._unused_device_archive_slots == list(range(archive_slots - 1, -1, -1))
    assert [tensor.data_ptr() for tensor in (states, *tier._device_archive, *tier._swap_tensors)] == pointers
    assert all(
        current is original for current, original in zip((states, *tier._device_archive, *tier._swap_tensors), storage)
    )
    assert tier._spill_count == spills
    assert tier.cache_status()["host_checkpoints"] == 0
    # Reused scheduler IDs must get fresh zero state, including an old spilled ID.
    mapped = tier.remap_table(np.array([[101, 108]], dtype=np.int32), 2)
    assert mapped.tolist() == [[1, 2]]
    assert torch.count_nonzero(states) == 0
    assert tier._restore_count == 0 and tier._spill_count == spills
    tier.reset()
    tier.reset()
    assert tier._unused_slots == [2, 1]
    assert len(set(tier._unused_device_archive_slots)) == archive_slots


def test_reset_synchronization_failure_preserves_checkpoint_metadata(monkeypatch):
    tier, _ = _tier()
    tier.remap_table(np.array([[101]], dtype=np.int32), 1)
    before = tier.cache_status()
    monkeypatch.setattr(tier, "_synchronize_device_state", Mock(side_effect=RuntimeError("pending writer")))
    with pytest.raises(RuntimeError, match="pending writer"):
        tier.reset()
    assert tier.cache_status() == before
    assert tier.slot_for(101) == 1


def test_spill_and_restore_preserves_prefix_checkpoint() -> None:
    tier, states = _tier()
    first = tier.remap_table(np.array([[0, 101, 102]], dtype=np.int32), 3)
    assert first.tolist() == [[0, 1, 2]]
    states[1].fill_(7)
    states[2].fill_(8)

    second = tier.remap_table(np.array([[0, 103, 102]], dtype=np.int32), 3)
    assert second.tolist() == [[0, 1, 2]]
    torch.testing.assert_close(states[2], torch.full((2,), 8, dtype=torch.float16))
    states[1].fill_(9)

    restored = tier.remap_table(np.array([[0, 101, 103]], dtype=np.int32), 3)
    assert restored.tolist() == [[0, 2, 1]]
    torch.testing.assert_close(states[2], torch.full((2,), 7, dtype=torch.float16))
    torch.testing.assert_close(states[1], torch.full((2,), 9, dtype=torch.float16))


def test_device_archive_defers_host_spill_and_preserves_lru_states() -> None:
    tier, states = _tier(archive_slots=2)
    mapped = tier.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
    states[mapped[0, 0]].fill_(7)
    states[mapped[0, 1]].fill_(8)

    tier.remap_table(np.array([[103, 102]], dtype=np.int32), 2)
    assert list(tier._device_archive_resident) == [101]
    assert not tier._host

    restored = tier.remap_table(np.array([[101, 103]], dtype=np.int32), 2)
    torch.testing.assert_close(states[restored[0, 0]], torch.full((2,), 7, dtype=torch.float16))
    assert list(tier._device_archive_resident) == [102]
    assert not tier._host

    tier.remap_table(np.array([[104, 103]], dtype=np.int32), 2)
    assert set(tier._device_archive_resident) == {101, 102}
    tier.remap_table(np.array([[105, 103]], dtype=np.int32), 2)
    assert set(tier._device_archive_resident) == {101, 104}
    assert set(tier._host) == {102}


def test_copy_on_write_uses_device_archive_before_host() -> None:
    tier, states = _tier(archive_slots=2)
    mapped = tier.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
    states[mapped[0, 0]].fill_(7)
    tier.remap_table(np.array([[103, 102]], dtype=np.int32), 2)

    tier.copy(101, 201)

    assert set(tier._device_archive_resident) == {101, 201}
    assert not tier._host
    restored = tier.remap_table(np.array([[201, 103]], dtype=np.int32), 2)
    torch.testing.assert_close(states[restored[0, 0]], torch.full((2,), 7, dtype=torch.float16))


def test_device_archive_budget_preserves_reserve_and_capacity_limit() -> None:
    assert prefix_mamba_device_archive_slots(10_000, 2_000, 100, 200) == 80
    assert prefix_mamba_device_archive_slots(10_000, 2_000, 100, 50) == 50
    assert prefix_mamba_device_archive_slots(1_000, 2_000, 100, 50) == 0
    with pytest.raises(ValueError, match="archive budget"):
        prefix_mamba_device_archive_slots(1_000, 0, 0, 50)


def test_device_archive_can_be_deferred_until_after_tier_creation() -> None:
    tier, states = _tier()
    assert not tier._device_archive
    assert not tier._swap_tensors

    tier.allocate_device_archive(2)

    assert tier.device_archive_slots == 2
    assert tier._device_archive[0].shape == (2, *states.shape[1:])
    with pytest.raises(RuntimeError, match="already allocated"):
        tier.allocate_device_archive(1)


def test_state_bytes_per_slot_includes_every_layer_and_state() -> None:
    fp16 = torch.zeros((3, 4), dtype=torch.float16)
    fp32 = torch.zeros((3, 2), dtype=torch.float32)
    assert prefix_mamba_state_bytes_per_slot([(fp16, fp32), (fp16,)]) == 24


def test_reused_block_id_discards_old_checkpoint() -> None:
    tier, states = _tier()
    tier.remap_table(np.array([[0, 101, 102]], dtype=np.int32), 3)
    states[1].fill_(7)
    tier.remap_table(np.array([[0, 103, 102]], dtype=np.int32), 3)
    tier.invalidate([101])
    tier.remap_table(np.array([[0, 101, 103]], dtype=np.int32), 3)
    torch.testing.assert_close(states[2], torch.zeros(2, dtype=torch.float16))


def test_copy_on_write_duplicates_state() -> None:
    tier, states = _tier()
    tier.remap_table(np.array([[0, 101]], dtype=np.int32), 2)
    states[1].fill_(5)
    tier.invalidate([102])
    tier.copy(101, 102)
    tier.remap_table(np.array([[0, 102]], dtype=np.int32), 2)
    torch.testing.assert_close(states[2], torch.full((2,), 5, dtype=torch.float16))


def test_excess_live_states_fails_instead_of_aliasing() -> None:
    tier, _ = _tier()
    with pytest.raises(RuntimeError, match="references 3 states"):
        tier.remap_table(np.array([[101, 102, 103]], dtype=np.int32), 3)


def test_postprocess_uses_compact_slots_not_global_block_223() -> None:
    tier, states = _tier()
    raw = np.array([[0, 223, 224]], dtype=np.int32)
    mapped = tier.remap_table(raw, 3)
    batch = SimpleNamespace(
        block_table=[SimpleNamespace(get_numpy_array=lambda: raw)],
        _prefix_mamba_postprocess_tables={0: mapped},
    )
    block_ids = get_mamba_postprocess_block_ids(batch, 0, 0)
    assert block_ids.tolist() == [0, 1, 2]
    assert states[block_ids].shape == (3, 2)
    assert raw.tolist() == [[0, 223, 224]]
    del batch._prefix_mamba_postprocess_tables
    assert get_mamba_postprocess_block_ids(batch, 0, 0).tolist() == [0, 223, 224]


@pytest.mark.parametrize("requests,drafts,expected", [(1, 2, 64), (2, 2, 64), (16, 2, 97), (32, 4, 321)])
def test_shared_slot_count_covers_previous_and_current_candidate_windows(requests, drafts, expected):
    assert prefix_mamba_slot_count(requests, drafts) == expected


@pytest.mark.parametrize("requests,drafts", [(0, 1), (1, -1)])
def test_invalid_slot_count_fails(requests, drafts):
    with pytest.raises(ValueError):
        prefix_mamba_slot_count(requests, drafts)


def test_active_windows_follow_each_requests_progress_and_speculation():
    assert prefix_mamba_active_columns([1026, 5], [130944, 256], [128, 3], 128, 2) == (
        (1022, 1023, 1024, 1025),
        (1, 2, 3, 4),
    )
    # A cached checkpoint can be far from the final prefill chunk destination.
    assert prefix_mamba_active_columns([54, 3], [23424, 0], [3783, 512], 512, 0) == ((45, 53), (0,))
    # Running state is ahead of computed progress after rejected draft tokens.
    assert prefix_mamba_active_columns([6], [255], [3], 128, 2, [3]) == ((2, 3, 4, 5),)


@pytest.mark.parametrize(
    "used,computed,scheduled,block_size,drafts,previous",
    [
        ([2], [0, 1], [1], 128, 0, None),
        ([2], [0], [0], 128, 0, None),
        ([2], [-1], [1], 128, 0, None),
        ([2], [0], [1], 0, 0, None),
        ([2], [0], [1], 128, -1, None),
        ([2], [0], [1], 128, 0, []),
    ],
)
def test_active_windows_reject_invalid_metadata(used, computed, scheduled, block_size, drafts, previous):
    with pytest.raises(ValueError):
        prefix_mamba_active_columns(used, computed, scheduled, block_size, drafts, previous)


def test_missing_destination_or_previous_state_is_not_silently_aliased():
    with pytest.raises(RuntimeError, match="destination"):
        prefix_mamba_active_columns([1], [128], [1], 128, 0)
    with pytest.raises(RuntimeError, match="previous"):
        prefix_mamba_active_columns([2], [128], [1], 128, 0, [2])


def test_uneven_rows_ignore_stale_padding_and_inactive_matching_ids():
    tier, _ = _tier(5)
    raw = np.array([[101, 102, 103, 104], [201, 202, 103, 999]], dtype=np.int32)
    original = raw.copy()
    mapped = tier.remap_rows(raw, [4, 2], [(2, 3), (0, 1)])
    assert mapped[0, :2].tolist() == [0, 0]
    assert mapped[1, 2:].tolist() == [0, 0]
    assert len(set(mapped[mapped > 0].tolist())) == 4
    np.testing.assert_array_equal(raw, original)


def test_all_requests_live_ids_are_protected_before_any_eviction():
    tier, states = _tier()
    tier.remap_table(np.array([[10, 20]], dtype=np.int32), 2)
    states[tier.slot_for(10)].fill_(10)
    states[tier.slot_for(20)].fill_(20)
    # Staging row zero alone would evict the oldest ID 10, needed by row one.
    mapped = tier.remap_rows(np.array([[30], [10]], dtype=np.int32), [1, 1], [(0,), (0,)])
    assert mapped[0, 0] != mapped[1, 0]
    torch.testing.assert_close(states[mapped[1, 0]], torch.full((2,), 10, dtype=torch.float16))
    torch.testing.assert_close(states[mapped[0, 0]], torch.zeros(2, dtype=torch.float16))


def test_reordering_and_finished_request_prefix_reuse_preserve_all_layer_states():
    conv = torch.zeros((3, 4, 2), dtype=torch.float16)
    temporal = torch.zeros((3, 2, 2), dtype=torch.float32)
    tier = PrefixMambaStateTier([(conv, temporal)], 3)
    table = np.array([[101], [201]], dtype=np.int32)
    mapped = tier.remap_rows(table, [1, 1], [(0,), (0,)])
    conv[mapped[0, 0]].fill_(11)
    temporal[mapped[0, 0]].fill_(12)
    conv[mapped[1, 0]].fill_(21)
    temporal[mapped[1, 0]].fill_(22)
    reordered = tier.remap_rows(table[::-1], [1, 1], [(0,), (0,)])
    np.testing.assert_array_equal(reordered, mapped[::-1])
    # Request 101 finishes; 201 is condensed into row zero and another arrives.
    tier.remap_rows(np.array([[201], [301]], dtype=np.int32), [1, 1], [(0,), (0,)])
    restored = tier.remap_rows(table, [1, 1], [(0,), (0,)])
    torch.testing.assert_close(conv[restored[0, 0]], torch.full((4, 2), 11, dtype=torch.float16))
    torch.testing.assert_close(temporal[restored[0, 0]], torch.full((2, 2), 12, dtype=torch.float32))
    torch.testing.assert_close(conv[restored[1, 0]], torch.full((4, 2), 21, dtype=torch.float16))
    torch.testing.assert_close(temporal[restored[1, 0]], torch.full((2, 2), 22, dtype=torch.float32))


def test_shared_prefix_cow_and_recycled_ids_do_not_contaminate_other_window():
    tier, states = _tier()
    tier.remap_rows(np.array([[101], [101]], dtype=np.int32), [1, 1], [(0,), (0,)])
    states[tier.slot_for(101)].fill_(7)
    tier.copy(101, 201)
    mapped = tier.remap_rows(np.array([[101], [201]], dtype=np.int32), [1, 1], [(0,), (0,)])
    states[mapped[1, 0]].fill_(9)
    torch.testing.assert_close(states[mapped[0, 0]], torch.full((2,), 7, dtype=torch.float16))
    tier.invalidate([201])
    torch.testing.assert_close(states[mapped[1, 0]], torch.zeros(2, dtype=torch.float16))
    torch.testing.assert_close(states[mapped[0, 0]], torch.full((2,), 7, dtype=torch.float16))


def test_over_capacity_or_invalid_rows_fail_before_evicting_existing_states():
    tier, states = _tier()
    tier.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
    states[1:].fill_(5)
    before = states.clone()
    resident = dict(tier._resident)
    with pytest.raises(RuntimeError, match="references 3 states"):
        tier.remap_rows(np.array([[201, 202], [301, 0]], dtype=np.int32), [2, 1], [(0, 1), (0,)])
    with pytest.raises(ValueError, match="window"):
        tier.remap_rows(np.array([[201, 202], [301, 0]], dtype=np.int32), [2, 1], [(0,), (1,)])
    assert dict(tier._resident) == resident
    torch.testing.assert_close(states, before)
