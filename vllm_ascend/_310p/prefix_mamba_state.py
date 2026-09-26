# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small NPU working set for align-mode Mamba prefix-cache states.

The scheduler uses a global block-ID pool for both attention and Mamba groups.
Only the current and checkpoint Mamba states need to be on the NPU, but cached
prefix checkpoints must remain recoverable after their NPU slots are reused.
This class keeps a persistent mapping for resident IDs and spills displaced
states to host tensors. Scheduler-owned block IDs are never modified.
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def get_mamba_postprocess_block_ids(input_batch: Any, group_id: int, req_idx: int) -> np.ndarray:
    """Use the same compact slots for align postprocess as for model forward.

    The scheduler-owned NumPy block table keeps global IDs. The 310P runner
    stages a separate compact table on the NPU for forward, then postprocess
    runs after forward has restored request metadata. It must read that staged
    mapping rather than index the compact state with a global block ID.
    """
    mapped_tables = getattr(input_batch, "_prefix_mamba_postprocess_tables", None)
    if mapped_tables is not None and group_id in mapped_tables:
        return mapped_tables[group_id][req_idx]
    return input_batch.block_table[group_id].get_numpy_array()[req_idx]


class PrefixMambaStateTier:
    """Map scheduler block IDs to a bounded set of device state slots."""

    def __init__(self, layer_states: Sequence[Sequence[torch.Tensor]], num_slots: int) -> None:
        if num_slots < 2:
            raise ValueError("Prefix Mamba state tier needs a null slot and at least one live slot")
        self.layer_states = tuple(tuple(states) for states in layer_states)
        self.num_slots = num_slots
        self._resident: OrderedDict[int, int] = OrderedDict()
        self._host: dict[int, tuple[torch.Tensor, ...]] = {}
        self._unused_slots = list(range(num_slots - 1, 0, -1))
        for states in self.layer_states:
            for state in states:
                if state.shape[0] != num_slots:
                    raise ValueError("All compact Mamba states must have the same slot count")
                state[0].zero_()

    def _slot_tensors(self, slot: int) -> tuple[torch.Tensor, ...]:
        return tuple(state[slot] for states in self.layer_states for state in states)

    def _snapshot(self, slot: int) -> tuple[torch.Tensor, ...]:
        return tuple(tensor.detach().to("cpu", copy=True) for tensor in self._slot_tensors(slot))

    def invalidate(self, block_ids: Sequence[int]) -> None:
        """Forget bytes belonging to newly allocated (possibly reused) IDs."""
        for block_id in block_ids:
            if block_id <= 0:
                continue
            self._host.pop(block_id, None)
            slot = self._resident.get(block_id)
            if slot is not None:
                for tensor in self._slot_tensors(slot):
                    tensor.zero_()

    def copy(self, source_id: int, target_id: int) -> None:
        """Apply a scheduler CoW copy to the tier's authoritative state."""
        if source_id <= 0 or target_id <= 0:
            return
        source_slot = self._resident.get(source_id)
        if source_slot is not None:
            snapshot = self._snapshot(source_slot)
        else:
            snapshot = self._host.get(source_id)
        if snapshot is None:
            raise RuntimeError(f"Mamba state copy source block {source_id} is not resident or spilled")
        self._host[target_id] = tuple(tensor.clone() for tensor in snapshot)
        target_slot = self._resident.get(target_id)
        if target_slot is not None:
            for target, source in zip(self._slot_tensors(target_slot), snapshot):
                target.copy_(source)

    def _admit(self, block_id: int, protected: set[int]) -> int:
        if (slot := self._resident.get(block_id)) is not None:
            self._resident.move_to_end(block_id)
            return slot
        if self._unused_slots:
            slot = self._unused_slots.pop()
        else:
            victim_id = next((old_id for old_id in self._resident if old_id not in protected), None)
            if victim_id is None:
                raise RuntimeError(f"Mamba prefix step needs more than {self.num_slots - 1} live state blocks")
            slot = self._resident.pop(victim_id)
            self._host[victim_id] = self._snapshot(slot)
        snapshot = self._host.pop(block_id, None)
        if snapshot is None:
            for tensor in self._slot_tensors(slot):
                tensor.zero_()
        else:
            for target, source in zip(self._slot_tensors(slot), snapshot):
                target.copy_(source)
        self._resident[block_id] = slot
        return slot

    def remap_table(
        self, table: np.ndarray, used_columns: int, active_columns: Sequence[int] | None = None
    ) -> np.ndarray:
        """Stage only the Mamba checkpoints read by this step.

        Align-mode kernels read the previous checkpoint and the destination
        block, not the entire sequence's historical block table. Historical
        entries can point at the null slot until a later prefix-cache hit
        makes one active again. The scheduler's CPU table remains unchanged.
        """
        if active_columns is None:
            active_columns = range(used_columns)
        columns = tuple(sorted(set(active_columns)))
        if not 0 <= used_columns <= table.shape[1] or any(not 0 <= col < used_columns for col in columns):
            raise ValueError("Invalid active Mamba block-table window")
        mapped = np.zeros_like(table)
        block_ids = {int(value) for value in np.unique(table[:, columns]) if value > 0}
        if len(block_ids) >= self.num_slots:
            raise RuntimeError(
                f"Mamba prefix table references {len(block_ids)} states but has "
                f"only {self.num_slots - 1} non-null NPU slots"
            )
        for block_id in sorted(block_ids):
            slot = self._admit(block_id, block_ids)
            # Compare against the immutable scheduler table: a mapped slot
            # number must not be mistaken for another scheduler block ID.
            mapped[table == block_id] = slot
        return mapped

    def slot_for(self, block_id: int) -> int:
        """Resolve a scheduler ID already staged by :meth:`remap_table`."""
        if block_id <= 0:
            return block_id
        slot = self._resident.get(block_id)
        if slot is None:
            raise RuntimeError(f"Mamba state block {block_id} was not staged")
        return slot
