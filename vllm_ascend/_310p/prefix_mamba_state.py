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
from collections.abc import Mapping, Sequence
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend._310p.transfer_audit import TransferLedger, copy_direction

PREFIX_MAMBA_MIN_SLOTS = 64


def supports_compact_live_mamba_state(max_num_reqs: int, model_type: str | None) -> bool:
    """Recognize models whose live recurrent state uses per-request slots."""
    return max_num_reqs == 1 or model_type in {"qwen4_exp_text", "glm5_next_text"}


def prefix_mamba_slot_count(max_num_reqs: int, num_speculative_tokens: int) -> int:
    """One shared pool: null slot plus both live windows of every request."""
    if max_num_reqs < 1 or num_speculative_tokens < 0:
        raise ValueError("Invalid compact Mamba request or speculation limit")
    return max(PREFIX_MAMBA_MIN_SLOTS, 1 + max_num_reqs * 2 * (1 + num_speculative_tokens))


def prefix_mamba_state_bytes_per_slot(layer_states: Sequence[Sequence[torch.Tensor]]) -> int:
    """Return the bytes needed to retain one checkpoint for a Mamba group."""
    return sum(state[0].numel() * state[0].element_size() for states in layer_states for state in states)


def prefix_mamba_device_archive_slots(
    free_device_bytes: int,
    reserve_bytes: int,
    bytes_per_checkpoint: int,
    maximum_slots: int,
) -> int:
    """Fill spare device memory with checkpoints before using host memory."""
    if free_device_bytes < 0 or reserve_bytes < 0 or bytes_per_checkpoint <= 0 or maximum_slots < 0:
        raise ValueError("Invalid prefix Mamba device archive budget")
    available_bytes = max(0, free_device_bytes - reserve_bytes)
    return min(maximum_slots, available_bytes // bytes_per_checkpoint)


class LiveMambaRequestSlots:
    """Stable per-request lanes when prefix caching is disabled.

    A lane holds only the current state and speculative candidates. Finished
    requests release their lane; row reordering never changes its owner. The
    caller zeros every state tensor in a newly assigned lane before reuse.
    """

    def __init__(self, max_requests: int, slots_per_request: int) -> None:
        if max_requests < 1 or slots_per_request < 1:
            raise ValueError("Live Mamba slots require positive request and slot counts")
        self.max_requests = max_requests
        self.slots_per_request = slots_per_request
        self._lanes: dict[str, int] = {}

    def assign(
        self, active_requests: Sequence[str], known_requests: set[str]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if len(active_requests) > self.max_requests or len(set(active_requests)) != len(active_requests):
            raise ValueError("Invalid live Mamba request batch")
        if not set(active_requests) <= known_requests:
            raise ValueError("Active Mamba request is missing from runner state")
        self._lanes = {req_id: lane for req_id, lane in self._lanes.items() if req_id in known_requests}
        free_lanes = sorted(set(range(self.max_requests)) - set(self._lanes.values()))
        newly_assigned = []
        for req_id in active_requests:
            if req_id not in self._lanes:
                if not free_lanes:
                    raise RuntimeError("No compact Mamba lane available for live request")
                lane = free_lanes.pop(0)
                self._lanes[req_id] = lane
                newly_assigned.append(lane)
        return tuple(self._lanes[req_id] for req_id in active_requests), tuple(newly_assigned)

    def mapped_columns(self, lanes: Sequence[int], columns: int) -> np.ndarray:
        if columns <= 0 or any(lane < 0 or lane >= self.max_requests for lane in lanes):
            raise ValueError("Invalid compact Mamba columns or lane")
        template = np.arange(columns, dtype=np.int32) % self.slots_per_request
        return np.stack([template + lane * self.slots_per_request for lane in lanes])


def prefix_mamba_active_columns(
    used_columns: Sequence[int],
    computed_tokens: Sequence[int],
    scheduled_tokens: Sequence[int],
    block_size: int,
    num_speculative_blocks: int,
    previous_columns: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Select each row's pre-copy sources and forward/postprocess destinations.

    The previous running column need not equal floor(computed / block_size)
    after speculative rejection. Protect its whole candidate window as well
    as the current one, so temporal-state copy can select the accepted token.
    All arguments are scheduler-side CPU metadata; no device synchronization.
    """
    num_reqs = len(used_columns)
    if (
        block_size <= 0
        or num_speculative_blocks < 0
        or len(computed_tokens) != num_reqs
        or len(scheduled_tokens) != num_reqs
        or (previous_columns is not None and len(previous_columns) != num_reqs)
    ):
        raise ValueError("Invalid per-request compact Mamba metadata")
    rows = []
    for row in range(num_reqs):
        computed, scheduled, used = int(computed_tokens[row]), int(scheduled_tokens[row]), int(used_columns[row])
        if computed < 0 or scheduled <= 0:
            raise ValueError("Compact Mamba requires nonnegative progress and a scheduled token")
        current = (computed + scheduled - 1) // block_size
        if current >= used:
            raise RuntimeError("Mamba destination block is missing from the scheduler table")
        previous = int(previous_columns[row]) if previous_columns is not None else (computed - 1) // block_size
        if previous < -1 or previous >= used:
            raise RuntimeError("Mamba previous state is missing from the scheduler table")
        needed = set(range(current, min(used, current + 1 + num_speculative_blocks)))
        if previous >= 0:
            needed.update(range(previous, min(used, previous + 1 + num_speculative_blocks)))
        rows.append(tuple(sorted(needed)))
    return tuple(rows)


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
    """Map scheduler block IDs to graph slots and a device archive."""

    def __init__(
        self,
        layer_states: Sequence[Sequence[torch.Tensor]],
        num_slots: int,
        device_archive_slots: int = 0,
    ) -> None:
        if num_slots < 2:
            raise ValueError("Prefix Mamba state tier needs a null slot and at least one live slot")
        if device_archive_slots < 0:
            raise ValueError("Prefix Mamba device archive slot count cannot be negative")
        self.layer_states = tuple(tuple(states) for states in layer_states)
        self.transfer_ledger = TransferLedger()
        self._state_device = next((state.device for states in self.layer_states for state in states), None)
        self.num_slots = num_slots
        self._resident: OrderedDict[int, int] = OrderedDict()
        self._device_archive_resident: OrderedDict[int, int] = OrderedDict()
        self._host: dict[int, tuple[torch.Tensor, ...]] = {}
        self._unused_slots = list(range(num_slots - 1, 0, -1))
        self._unused_device_archive_slots: list[int] = []
        self._spill_count = 0
        self._restore_count = 0
        self._device_archive_hit_count = 0
        self._retirement_count = 0
        self._bytes_per_slot = prefix_mamba_state_bytes_per_slot(self.layer_states)
        self._device_archive: tuple[torch.Tensor, ...] = ()
        self._swap_tensors: tuple[torch.Tensor, ...] = ()
        self.device_archive_slots = 0
        for states in self.layer_states:
            for state in states:
                if state.shape[0] != num_slots:
                    raise ValueError("All compact Mamba states must have the same slot count")
                state[0].zero_()
        if device_archive_slots:
            self.allocate_device_archive(device_archive_slots)

    def allocate_device_archive(self, num_slots: int) -> None:
        """Allocate the second device tier after graph/HCCL initialization."""
        if num_slots < 0:
            raise ValueError("Prefix Mamba device archive slot count cannot be negative")
        if self.device_archive_slots:
            raise RuntimeError("Prefix Mamba device archive is already allocated")
        if not num_slots:
            return
        if self._device_archive_resident or self._host:
            raise RuntimeError("Prefix Mamba device archive must be allocated before serving requests")
        self._device_archive = tuple(
            torch.empty(
                (num_slots, *state.shape[1:]),
                dtype=state.dtype,
                device=state.device,
            )
            for states in self.layer_states
            for state in states
        )
        self._swap_tensors = tuple(torch.empty_like(state[0]) for states in self.layer_states for state in states)
        self._unused_device_archive_slots = list(range(num_slots - 1, -1, -1))
        self.device_archive_slots = num_slots

    def _slot_tensors(self, slot: int) -> tuple[torch.Tensor, ...]:
        return tuple(state[slot] for states in self.layer_states for state in states)

    def _device_archive_tensors(self, slot: int) -> tuple[torch.Tensor, ...]:
        return tuple(state[slot] for state in self._device_archive)

    def _synchronize_device_state(self) -> None:
        # A previous forward can leave a state write queued on another NPU
        # stream.  The tier must not archive or overwrite that slot until the
        # write completes; synchronizing only the current stream is insufficient.
        if self._state_device is not None and self._state_device.type == "npu":
            torch.npu.synchronize(self._state_device)

    def _drain(self, reason: str) -> None:
        started = perf_counter_ns()
        self._synchronize_device_state()
        if self._state_device is not None and self._state_device.type == "npu":
            self.transfer_ledger.record("barrier", reason, elapsed_ns=perf_counter_ns() - started)

    def reset(self) -> None:
        """Forget every checkpoint after the scheduler cache has been drained.

        Keep primary, archive and swap storage intact for captured graphs.
        Reclaimed primary slots are zeroed by their next admission. Counters
        remain cumulative so callers can compare transfer deltas across resets.
        This must only be called with no active or pending model execution.
        """
        self._drain("reset")
        self._resident.clear()
        self._device_archive_resident.clear()
        self._host.clear()
        self._unused_slots = list(range(self.num_slots - 1, 0, -1))
        self._unused_device_archive_slots = list(range(self.device_archive_slots - 1, -1, -1))

    def cache_status(self) -> dict[str, int]:
        """Report host metadata and cumulative transfers without device reads."""
        return {
            "resident_checkpoints": len(self._resident),
            "archive_checkpoints": len(self._device_archive_resident),
            "host_checkpoints": len(self._host),
            "primary_slots": self.num_slots - 1,
            "archive_slots": self.device_archive_slots,
            "spill_count": self._spill_count,
            "restore_count": self._restore_count,
            "device_archive_hit_count": self._device_archive_hit_count,
            "retirement_count": self._retirement_count,
        }

    def retain_blocks(self, block_ids: Sequence[int], *, synchronized: bool = False) -> None:
        """Retire only states the scheduler proves uncached and unowned.

        The bounded scheduler includes request-owned blocks and pending CoW
        sources in its snapshot. Independently dropping a worker's LRU state
        would leave a valid prefix hash pointing at missing recurrent bytes.
        Storage and the null slot stay fixed for already captured graphs.
        """
        retained = set(block_ids)
        resident = set(self._resident) - retained
        archived = set(self._device_archive_resident) - retained
        host = set(self._host) - retained
        if (resident or archived) and not synchronized:
            # A previous graph may still be writing on a different stream.
            # Drain before reusing its slots, and leave metadata intact if
            # synchronization fails. No device reads or host snapshot copies.
            self._drain("retirement")
        for block_id in resident:
            self._unused_slots.append(self._resident.pop(block_id))
        for block_id in archived:
            self._release_device_archive(block_id)
        for block_id in host:
            del self._host[block_id]
        self._retirement_count += len(resident) + len(archived) + len(host)

    def _copy_tensors(
        self, targets: Sequence[torch.Tensor], sources: Sequence[torch.Tensor], reason: str = "checkpoint_copy"
    ) -> None:
        for target, source in zip(targets, sources):
            started = perf_counter_ns()
            target.copy_(source)
            elapsed = perf_counter_ns() - started
            self.transfer_ledger.record(
                copy_direction(source.device, target.device),
                reason,
                nbytes=target.numel() * target.element_size(),
                elapsed_ns=elapsed,
            )

    @staticmethod
    def _snapshot(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(tensor.detach().to("cpu", copy=True) for tensor in tensors)

    def _release_device_archive(self, block_id: int) -> int | None:
        slot = self._device_archive_resident.pop(block_id, None)
        if slot is not None:
            self._unused_device_archive_slots.append(slot)
        return slot

    def _spill_to_host(self, block_id: int, tensors: Sequence[torch.Tensor]) -> None:
        started = perf_counter_ns()
        self._host[block_id] = self._snapshot(tensors)
        elapsed = perf_counter_ns() - started
        for source, target in zip(tensors, self._host[block_id]):
            self.transfer_ledger.record(
                copy_direction(source.device, target.device),
                "host_spill",
                nbytes=target.numel() * target.element_size(),
                elapsed_ns=elapsed,
            )
            elapsed = 0  # Charge the whole snapshot once, not once per tensor.
        self._spill_count += 1
        if self._spill_count == 1:
            logger.warning(
                "Prefix Mamba device tiers exhausted: primary_slots=%d, "
                "archive_slots=%d, checkpoint_bytes=%d. Spilling checkpoints "
                "NPU->CPU; subsequent prefix hits may restore CPU->NPU.",
                self.num_slots - 1,
                self.device_archive_slots,
                self._bytes_per_slot,
            )

    def _store_on_device_or_host(
        self,
        block_id: int,
        tensors: Sequence[torch.Tensor],
        protected: set[int],
    ) -> None:
        if self._unused_device_archive_slots:
            archive_slot = self._unused_device_archive_slots.pop()
        else:
            archive_victim = next(
                (old_id for old_id in self._device_archive_resident if old_id not in protected),
                None,
            )
            if archive_victim is None:
                self._spill_to_host(block_id, tensors)
                return
            archive_slot = self._device_archive_resident.pop(archive_victim)
            self._spill_to_host(archive_victim, self._device_archive_tensors(archive_slot))
        self._copy_tensors(self._device_archive_tensors(archive_slot), tensors, "archive_store")
        self._device_archive_resident[block_id] = archive_slot

    def invalidate(self, block_ids: Sequence[int], *, synchronized: bool = False) -> None:
        """Forget bytes belonging to newly allocated (possibly reused) IDs."""
        if not synchronized and any(block_id > 0 and block_id in self._resident for block_id in block_ids):
            self._drain("invalidation")
        for block_id in block_ids:
            if block_id <= 0:
                continue
            self._host.pop(block_id, None)
            self._release_device_archive(block_id)
            slot = self._resident.get(block_id)
            if slot is not None:
                for tensor in self._slot_tensors(slot):
                    tensor.zero_()

    def copy(self, source_id: int, target_id: int, *, synchronized: bool = False) -> None:
        """Apply a scheduler CoW copy to the tier's authoritative state."""
        if source_id <= 0 or target_id <= 0:
            return
        if not synchronized and (source_id in self._resident or target_id in self._resident):
            self._drain("copy_on_write")
        source_slot = self._resident.get(source_id)
        if source_slot is not None:
            source = self._slot_tensors(source_slot)
        elif (archive_slot := self._device_archive_resident.get(source_id)) is not None:
            self._device_archive_resident.move_to_end(source_id)
            source = self._device_archive_tensors(archive_slot)
        else:
            source = self._host.get(source_id)
        if source is None:
            raise RuntimeError(f"Mamba state copy source block {source_id} is not resident or spilled")
        self._host.pop(target_id, None)
        target_slot = self._resident.get(target_id)
        if target_slot is not None:
            self._release_device_archive(target_id)
            self._copy_tensors(self._slot_tensors(target_slot), source, "copy_on_write")
            return
        archive_slot = self._device_archive_resident.get(target_id)
        if archive_slot is not None:
            self._device_archive_resident.move_to_end(target_id)
            self._copy_tensors(self._device_archive_tensors(archive_slot), source, "copy_on_write")
            return
        self._store_on_device_or_host(target_id, source, {source_id})

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
            incoming_archive_slot = self._device_archive_resident.pop(block_id, None)
            if incoming_archive_slot is not None:
                incoming = self._device_archive_tensors(incoming_archive_slot)
                self._copy_tensors(self._swap_tensors, incoming, "archive_swap_stage")
                self._copy_tensors(incoming, self._slot_tensors(slot), "archive_swap_out")
                self._copy_tensors(self._slot_tensors(slot), self._swap_tensors, "archive_swap_in")
                self._device_archive_resident[victim_id] = incoming_archive_slot
                self._device_archive_hit_count += 1
                self._resident[block_id] = slot
                return slot
            self._store_on_device_or_host(victim_id, self._slot_tensors(slot), protected)
        archive_slot = self._device_archive_resident.pop(block_id, None)
        if archive_slot is not None:
            snapshot = self._device_archive_tensors(archive_slot)
            self._copy_tensors(self._slot_tensors(slot), snapshot, "checkpoint_restore")
            self._unused_device_archive_slots.append(archive_slot)
            self._device_archive_hit_count += 1
            self._resident[block_id] = slot
            return slot
        snapshot = self._host.pop(block_id, None)
        if snapshot is None:
            for tensor in self._slot_tensors(slot):
                tensor.zero_()
        else:
            self._restore_count += 1
            if self._restore_count == 1:
                logger.warning(
                    "Restoring a spilled prefix Mamba checkpoint CPU->NPU: checkpoint_bytes=%d, spill_count=%d.",
                    self._bytes_per_slot,
                    self._spill_count,
                )
            self._copy_tensors(self._slot_tensors(slot), snapshot, "checkpoint_restore")
        self._resident[block_id] = slot
        return slot

    def remap_table(
        self, table: np.ndarray, used_columns: int, active_columns: Sequence[int] | None = None
    ) -> np.ndarray:
        """Compatibility entry point when every request uses the same columns."""
        columns = tuple(range(used_columns) if active_columns is None else active_columns)
        return self.remap_rows(table, [used_columns] * len(table), [columns] * len(table))

    def remap_rows(
        self,
        table: np.ndarray,
        used_columns: Sequence[int],
        active_columns: Sequence[Sequence[int]],
        *,
        synchronized: bool = False,
    ) -> np.ndarray:
        """Stage only the Mamba checkpoints read by this step.

        Align-mode kernels read the previous checkpoint and the destination
        block, not the entire sequence's historical block table. Historical
        entries can point at the null slot until a later prefix-cache hit
        makes one active again. The scheduler's CPU table remains unchanged.

        Each row may have a different length/progress. Admit the UNION first:
        admitting rows separately can evict the first request's live state
        while staging the second request, silently aliasing their outputs.
        """
        if table.ndim != 2 or len(used_columns) != len(table) or len(active_columns) != len(table):
            raise ValueError("Invalid per-request Mamba block-table shape")
        rows = [tuple(sorted(set(columns))) for columns in active_columns]
        block_ids: set[int] = set()
        for row, (used, columns) in enumerate(zip(used_columns, rows)):
            if not 0 <= used <= table.shape[1] or any(not 0 <= col < used for col in columns):
                raise ValueError("Invalid active Mamba block-table window")
            block_ids.update(int(value) for value in table[row, list(columns)] if value > 0)
        mapped = np.zeros_like(table)
        if len(block_ids) >= self.num_slots:
            raise RuntimeError(
                f"Mamba prefix table references {len(block_ids)} states but has "
                f"only {self.num_slots - 1} non-null NPU slots"
            )
        if not synchronized and len(block_ids.difference(self._resident)) > len(self._unused_slots):
            self._drain("admission")
        for block_id in sorted(block_ids):
            self._admit(block_id, block_ids)
        for row, columns in enumerate(rows):
            for column in columns:
                block_id = int(table[row, column])
                mapped[row, column] = self._resident[block_id] if block_id > 0 else 0
        return mapped

    def slot_for(self, block_id: int) -> int:
        """Resolve a scheduler ID already staged by :meth:`remap_table`."""
        if block_id <= 0:
            return block_id
        slot = self._resident.get(block_id)
        if slot is None:
            raise RuntimeError(f"Mamba state block {block_id} was not staged")
        return slot


def retain_prefix_mamba_blocks(
    tiers: Mapping[int, PrefixMambaStateTier] | None,
    retained_ids: Mapping[int, Sequence[int]],
    *,
    already_synchronized: bool = False,
) -> None:
    """Retire all groups after one worker-wide drain, preserving graph storage.

    Every tier in a worker uses its one NPU. Retirement only updates host
    metadata, so no work is enqueued between the drain and group reclamation.
    Reuse the runner's layout-change drain when it has already completed.
    """
    if tiers is None or retained_ids.keys() != tiers.keys():
        raise RuntimeError("Bounded Mamba checkpoint snapshot does not match worker cache groups")
    retained = {group_id: set(block_ids) for group_id, block_ids in retained_ids.items()}
    needs_drain = any(
        tier._resident.keys() - retained[group_id] or tier._device_archive_resident.keys() - retained[group_id]
        for group_id, tier in tiers.items()
    )
    if needs_drain and not already_synchronized:
        next(iter(tiers.values()))._drain("batch_retirement")
    for group_id, tier in tiers.items():
        tier.retain_blocks(retained_ids[group_id], synchronized=True)


def apply_prefix_mamba_updates(tiers, fresh_ids, copies, *, batched: bool = False) -> None:
    """Run one serialized invalidation/CoW phase after base runner updates.

    All copies/zeros here enqueue on the caller's current stream. A fresh
    worker-wide drain protects primary-slot writes in this phase; a drain
    from before base runner updates must not be reused. Archive storage is
    only touched by serialized current-stream copies, not model execution.
    No model work may interleave with this call.
    This is ordering consolidation, not an atomic rollback transaction.
    """
    if set(fresh_ids) != set(tiers) or set(copies) != set(tiers):
        raise ValueError("Mamba update groups must match the resident tiers")
    needs_drain = any(
        any(block_id in tier._resident for block_id in fresh_ids[group])
        or any(
            source > 0 and target > 0 and (source in tier._resident or target in tier._resident)
            for source, target in copies[group]
        )
        for group, tier in tiers.items()
    )
    if batched and needs_drain:
        _drain_prefix_phase(tiers, "batch_updates")
    for group, tier in tiers.items():
        tier.invalidate(fresh_ids[group], synchronized=batched)
        for source_id, target_id in copies[group]:
            tier.copy(source_id, target_id, synchronized=batched)


def _drain_prefix_phase(tiers, reason):
    devices = {tier._state_device for tier in tiers.values()}
    if len(devices) > 1:
        raise ValueError("A prefix phase must use one worker device")
    if tiers:
        # The selected phase needs primary storage reuse. Drain the whole
        # worker once so previous model writers on every stream are complete.
        next(iter(tiers.values()))._drain(reason)


def remap_prefix_mamba_rows(tiers, plans, *, batched: bool = False):
    """Admit all groups before staging device tables, with one phase drain.

    Plans contain CPU tables, used-column counts and active windows. The union
    of each group's live IDs is still admitted by remap_rows; groups never
    share storage. No unrelated device writes are submitted inside this call.
    """
    if set(plans) != set(tiers):
        raise ValueError("Mamba remap plans must match all worker tiers")
    if batched:
        # Admission hits that neither overwrite nor move state require no drain.
        needs_drain = any(
            len(
                {int(table[row, col]) for row, columns in enumerate(active) for col in columns if table[row, col] > 0}
                - tier._resident.keys()
            )
            > len(tier._unused_slots)
            for group, tier in tiers.items()
            for table, used, active in (plans[group],)
        )
        if needs_drain:
            _drain_prefix_phase(tiers, "batch_admission")
    return {group: tier.remap_rows(*plans[group], synchronized=batched) for group, tier in tiers.items()}
