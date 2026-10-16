# SPDX-License-Identifier: Apache-2.0
"""Unbound QSA owner-selection experiment; existing decode graphs stay intact.

The caller updates every rank's index cache before entering this schedule.
Indexer callbacks must not mutate shared RoPE memoization while local Q/K
preprocessing runs. Binding this to a model requires cross-rank selection
parity, cache-reuse, and end-to-end timing qualification first.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection, copy_group_selection_into

DEFAULT_MIN_OVERLAP_TOKENS = 128
SELECTION_METADATA_COLUMNS = 3


@dataclass(frozen=True)
class OwnerSelectionPolicy:
    rank: int
    world_size: int
    owner: int
    min_tokens: int = DEFAULT_MIN_OVERLAP_TOKENS

    def __post_init__(self):
        if (
            self.world_size <= 0
            or not 0 <= self.rank < self.world_size
            or not 0 <= self.owner < self.world_size
            or self.min_tokens <= 0
        ):
            raise ValueError("invalid QSA owner selection policy")

    def allows_overlap(self, *, tokens: int, captured: bool, pure_prefill: bool) -> bool:
        # These scheduler-owned admission fields must agree on all ranks.
        # Mixed prefill/decode stays synchronous until it has its own gate.
        return self.world_size > 1 and tokens >= self.min_tokens and pure_prefill and not captured


def selection_packet(tokens: int, groups: int, device: torch.device) -> tuple[torch.Tensor, QSAGroupSelection]:
    """One compact INT32 broadcast; each attention operand remains contiguous."""
    if tokens <= 0 or groups <= 0:
        raise ValueError("positive QSA packet geometry required")
    packet = torch.empty(tokens * (groups + SELECTION_METADATA_COLUMNS), dtype=torch.int32, device=device)
    indices_end = tokens * groups
    selection = QSAGroupSelection(
        packet[:indices_end].view(tokens, groups),
        packet[indices_end : indices_end + tokens],
        packet[indices_end + tokens : indices_end + 2 * tokens],
        packet[indices_end + 2 * tokens :],
    )
    return packet, selection


class QSAOwnerOverlap:
    """Enqueue owner selection and its broadcast beside local Q/K preparation.

    ``broadcast`` must enqueue an in-place collective on the supplied stream,
    use group-relative ``owner``, and must never silently skip a disabled
    communicator. All ranks use the same collective order and policy. No rank
    may retry via the replicated path once the collective is submitted.

    ``npu`` is torch.npu in a worker, explicitly injected for offline protocol
    tests. This resource owns its stream; no process-global stream state is
    added. The runtime path uses events rather than a host synchronization.
    """

    def __init__(self, policy: OwnerSelectionPolicy, broadcast: Callable, *, npu: Any):
        self.policy = policy
        self.broadcast = broadcast
        self.npu = npu
        self.selection_stream = npu.Stream()

    def run(
        self,
        *,
        tokens: int,
        groups: int,
        device: torch.device,
        captured: bool,
        pure_prefill: bool,
        indexer_inputs: Sequence[torch.Tensor],
        select: Callable[[], QSAGroupSelection],
        prepare_local: Callable[[], Any],
    ) -> tuple[Any, QSAGroupSelection]:
        if not self.policy.allows_overlap(tokens=tokens, captured=captured, pure_prefill=pure_prefill):
            return prepare_local(), select()
        if any(tensor.device != device for tensor in indexer_inputs):
            raise ValueError("QSA indexer inputs must be on the local selection device")
        # Allocate on the consumer stream and explicitly retain across the
        # producer's asynchronous use. Each call gets a distinct live packet.
        main = self.npu.current_stream(device)
        packet, selection = selection_packet(tokens, groups, device)
        ready, done = self.npu.Event(), self.npu.Event()
        ready.record(main)
        packet.record_stream(self.selection_stream)
        for tensor in indexer_inputs:
            tensor.record_stream(self.selection_stream)
        with self.npu.stream(self.selection_stream):
            self.selection_stream.wait_event(ready)
            if self.policy.rank == self.policy.owner:
                copy_group_selection_into(selection, select())
            self.broadcast(packet, self.policy.owner, self.selection_stream)
            done.record(self.selection_stream)
        local = prepare_local()
        # The caller can only consume selection after this device-side join.
        main.wait_event(done)
        return local, selection
