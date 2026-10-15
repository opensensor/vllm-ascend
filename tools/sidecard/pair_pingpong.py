# SPDX-License-Identifier: Apache-2.0
"""Pure policy for a two-inbox/two-READY push SUM experiment; no device code.

For sequence s, each rank enqueues on ONE owning in-order stream:
producer(s); push input into peer.inbox[s%2]; record peer.READY[s%2];
wait own.READY[s%2]; reset own.READY[s%2]; SUM; consumer(s).
The next sequence starts after that consumer. Every push byte completes before
its READY record, and SUM reads all input bytes before completing. Both ranks
execute the same monotonic sequence and both original reductions per layer.

Reuse proof: A cannot begin push(s+2) until its wait for B.READY(s+1) completed.
B records READY(s+1) only AFTER its wait/reset/SUM/consumer(s). Thus B has both
read A's old inbox slot s%2 and reset A's old READY slot s%2 before A can write
or record there again. This argument applies to every byte/suboperation, not
only whole-copy or whole-reduction milestones. The other slot protects the
intervening sequence; a single slot has no corresponding acknowledgement.

Prerequisites, NOT established here: shareable binary Notify release/acquire
and reset semantics, visibility of completed peer SDMA writes, strict stream
ordering (including all producer/consumer work), disjoint inbox allocations,
and the original input/output aliasing contract. This proof covers PUSH only;
it does not qualify the existing remote-read mode, cross-stream consumers,
out-of-order calls, cancellation, failed peers or uncertain DMA quiescence.

Local drain does not prove peer drain. Keep the existing both-peer drain and
import-release/export-close ownership protocol, bounded host submission window,
and monotonic sequence across drain windows/requests. A fresh session starts
with clear READY objects. Overflow requires a new fully drained session.
CPU policy/simulation is not NPU qualification, graph safety or a speed claim.
"""

from dataclasses import dataclass

from .pair_collective import BOUNDARIES, MAX_PENDING, MAX_ROWS, WIDTH

SLOTS = 2
READY_OBJECTS = 2
MAX_SEQUENCE = (1 << 64) - 1
FLOAT32_BYTES = 4


def slot_for_sequence(sequence):
    """Select a slot without resetting parity at local drain/request boundaries."""
    if type(sequence) is not int or not 0 <= sequence <= MAX_SEQUENCE:
        raise ValueError("sequence must be an unsigned 64-bit integer; wrap requires a new drained session")
    return sequence & 1


@dataclass(frozen=True)
class PingPongStep:
    sequence: int
    slot: int
    boundary: str


@dataclass(frozen=True)
class PingPongPolicy:
    """Bounded immutable admission policy; the caller owns monotonic state.

    No enqueue, allocation, Notify, stream, reset, teardown or fallback occurs
    here. Both ranks must admit the same step before native submission.
    """

    rows: int = 1
    max_pending: int = MAX_PENDING

    def __post_init__(self):
        if type(self.rows) is not int or not 1 <= self.rows <= MAX_ROWS:
            raise ValueError("fixed-shape lane rows must be from 1 through 32")
        if type(self.max_pending) is not int or not 1 <= self.max_pending <= MAX_PENDING:
            raise ValueError("pending host window must retain the existing bounded admission limit")

    @property
    def inbox_bytes(self):
        return SLOTS * self.rows * WIDTH * FLOAT32_BYTES

    def step(self, sequence, boundary, *, expected_sequence, pending):
        slot = slot_for_sequence(sequence)
        if type(expected_sequence) is not int or sequence != expected_sequence:
            raise ValueError("collective sequence is stale or out of order")
        if boundary != BOUNDARIES[sequence % len(BOUNDARIES)]:
            raise ValueError("both attention and MLP reductions must retain their agreed layer order")
        if type(pending) is not int or not 0 <= pending < self.max_pending:
            raise ValueError("drain the bounded host submission window before enqueueing more")
        return PingPongStep(sequence, slot, boundary)
