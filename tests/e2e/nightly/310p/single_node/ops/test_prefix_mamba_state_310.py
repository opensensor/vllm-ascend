# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual NPU copies/spills for interleaved compact state windows (not model accuracy)."""

import numpy as np
import pytest
import torch

from vllm_ascend._310p.prefix_mamba_state import PrefixMambaStateTier

NUM_SLOTS = 64
NUM_STEPS = 96
ID_STRIDE = 1000
STREAM_DELAY_MATMUL_SIZE = 8192
STREAM_DELAY_REPEATS = 5


@pytest.mark.parametrize("num_windows", [2, 4])
@torch.inference_mode()
def test_interleaved_npu_windows_spill_restore_cow_and_recycle(num_windows):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    states = (
        torch.zeros((NUM_SLOTS, 4, 128), dtype=torch.float16, device="npu"),
        torch.zeros((NUM_SLOTS, 2, 32, 32), dtype=torch.float32, device="npu"),
    )
    tier = PrefixMambaStateTier([states], NUM_SLOTS)
    pointers = [state.data_ptr() for state in states]
    for step in range(1, NUM_STEPS + 1):
        raw = np.array(
            [[ID_STRIDE * (window + 1) + step - 1, ID_STRIDE * (window + 1) + step] for window in range(num_windows)],
            dtype=np.int32,
        )
        mapped = tier.remap_rows(raw, [2] * num_windows, [(0, 1)] * num_windows)
        assert len(set(mapped.flat)) == num_windows * 2
        for window, (previous, current) in enumerate(mapped):
            for state in states:
                # Every window has its own recurrence; no device .item() reads.
                state[current].copy_(state[previous] + window + 1)
    torch.npu.synchronize()
    assert tier._host  # More historical checkpoints than resident slots.
    assert len(tier._resident) == NUM_SLOTS - 1
    assert [state.data_ptr() for state in states] == pointers

    # Alternate old and recent checkpoints, reversing request rows each time.
    for step in (1, NUM_STEPS, 7, 64, 32):
        windows = list(reversed(range(num_windows)))
        raw = np.array([[ID_STRIDE * (window + 1) + step] for window in windows], dtype=np.int32)
        mapped = tier.remap_rows(raw, [1] * num_windows, [(0,)] * num_windows)
        for row, window in enumerate(windows):
            for state in states:
                actual = state[mapped[row, 0]].cpu()
                torch.testing.assert_close(actual, torch.full_like(actual, step * (window + 1)), rtol=0, atol=0)

    source_id, target_id = ID_STRIDE + 1, ID_STRIDE * (num_windows + 1)
    tier.invalidate([target_id])
    tier.copy(source_id, target_id)  # Source can be spilled, not necessarily resident.
    mapped = tier.remap_rows(np.array([[source_id], [target_id]], dtype=np.int32), [1, 1], [(0,), (0,)])
    for state in states:
        torch.testing.assert_close(state[mapped[0, 0]].cpu(), state[mapped[1, 0]].cpu(), rtol=0, atol=0)
        state[mapped[1, 0]].fill_(99)
    tier.invalidate([target_id])
    for state in states:
        source, target = state[mapped[0, 0]].cpu(), state[mapped[1, 0]].cpu()
        torch.testing.assert_close(source, torch.ones_like(source), rtol=0, atol=0)
        torch.testing.assert_close(target, torch.zeros_like(target), rtol=0, atol=0)


@torch.inference_mode()
def test_spill_waits_for_state_write_on_another_npu_stream():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    state = torch.zeros((2, 1024 * 1024), dtype=torch.float16, device="npu")
    tier = PrefixMambaStateTier([(state,)], 2)
    tier.remap_table(np.array([[101]], dtype=np.int32), 1)

    matrix = torch.randn(
        (STREAM_DELAY_MATMUL_SIZE, STREAM_DELAY_MATMUL_SIZE),
        dtype=torch.float16,
        device="npu",
    )
    matrix @ matrix
    tier.remap_table(np.array([[202]], dtype=np.int32), 1)
    tier.remap_table(np.array([[101]], dtype=np.int32), 1)
    state[tier.slot_for(101)].zero_()
    # Warm both the kernel and CPU snapshot/restore before timing the race.
    torch.npu.synchronize()
    side_stream = torch.npu.Stream(device=0)
    with torch.npu.stream(side_stream):
        for _ in range(STREAM_DELAY_REPEATS):
            matrix @ matrix
        state[tier.slot_for(101)].fill_(42)

    # The host snapshot used to complete before the side-stream write.
    tier.remap_table(np.array([[303]], dtype=np.int32), 1)
    torch.npu.synchronize()
    restored = tier.remap_table(np.array([[101]], dtype=np.int32), 1)
    actual = state[restored[0, 0]].cpu()
    torch.testing.assert_close(actual, torch.full_like(actual, 42), rtol=0, atol=0)
