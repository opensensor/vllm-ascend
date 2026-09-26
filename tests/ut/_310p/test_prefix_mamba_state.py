# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p.prefix_mamba_state import PrefixMambaStateTier, get_mamba_postprocess_block_ids


def _tier(num_slots: int = 3) -> tuple[PrefixMambaStateTier, torch.Tensor]:
    states = torch.zeros((num_slots, 2), dtype=torch.float16)
    return PrefixMambaStateTier([(states,)], num_slots), states


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
