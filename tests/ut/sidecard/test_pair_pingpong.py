# SPDX-License-Identifier: Apache-2.0
"""Adversarial byte/suboperation model; this is not accelerator validation."""

import random
import struct
from collections import deque

import pytest

from tools.sidecard.pair_collective import BOUNDARIES, MAX_PENDING
from tools.sidecard.pair_pingpong import MAX_SEQUENCE, PingPongPolicy, slot_for_sequence

LANES = 3
WORD_BYTES = 4
PAYLOAD_BYTES = LANES * WORD_BYTES


class Hazard(AssertionError):
    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


def input_values(rank, sequence):
    # Nonzero, changing, exactly representable FP32 inputs and exact sums.
    return tuple(float((sequence + 1) * (17 + rank * 12) + lane * (rank + 3)) for lane in range(LANES))


def packed_input(rank, sequence):
    return struct.pack("<" + "f" * LANES, *input_values(rank, sequence))


class ByteFabric:
    """Two independent in-order streams; arbitrary cross-stream interleavings.

    Copy reads/writes, reduction byte reads/output writes, Notify wait/reset,
    producers and consumers are independently scheduled suboperations. Generation
    tags accompany every byte, detecting ABA even when float bytes coincide.
    """

    def __init__(self, slots=2, *, fail_fast=True):
        self.slots, self.fail_fast = slots, fail_fast
        self.queues = (deque(), deque())
        self.input_bytes = [[None] * PAYLOAD_BYTES for _ in range(2)]
        self.input_generations = [[None] * PAYLOAD_BYTES for _ in range(2)]
        self.inboxes = [[[None] * PAYLOAD_BYTES for _ in range(slots)] for _ in range(2)]
        self.inbox_generations = [[[None] * PAYLOAD_BYTES for _ in range(slots)] for _ in range(2)]
        self.consumed = [[[True] * PAYLOAD_BYTES for _ in range(slots)] for _ in range(2)]
        self.ready = [[None] * slots for _ in range(2)]
        self.output = [[None] * LANES for _ in range(2)]
        self.output_generations = [[None] * LANES for _ in range(2)]
        self.observed, self.violations, self.trace = ([], []), [], []
        self.steps, self.next_sequence = 0, [0, 0]

    def check(self, condition, kind, detail):
        if not condition:
            self.violations.append(kind)
            if self.fail_fast:
                raise Hazard(kind, detail)

    def enqueue(self, rank, sequence):
        assert sequence == self.next_sequence[rank]
        self.next_sequence[rank] += 1
        peer, slot = 1 - rank, sequence % self.slots
        expected_bytes = packed_input(rank, sequence)
        copy_bytes, read_own, read_peer = {}, {}, {}
        consumed_output, waited = [], []
        queue = self.queues[rank]

        def operation(name, action, predicate=lambda: True):
            queue.append((predicate, action, f"rank{rank}:{sequence}:{name}"))

        def produce(byte):
            self.input_bytes[rank][byte] = expected_bytes[byte]
            self.input_generations[rank][byte] = sequence

        def copy_read(byte):
            self.check(self.input_generations[rank][byte] == sequence, "source_generation", (rank, sequence, byte))
            copy_bytes[byte] = self.input_bytes[rank][byte]

        def copy_write(byte):
            self.check(self.consumed[peer][slot][byte], "inbox_overwrite", (rank, sequence, slot, byte))
            self.inboxes[peer][slot][byte] = copy_bytes[byte]
            self.inbox_generations[peer][slot][byte] = sequence
            self.consumed[peer][slot][byte] = False

        def record():
            self.check(self.ready[peer][slot] is None, "notify_coalescing", (rank, sequence, slot))
            # A second record on a binary object cannot queue another wakeup.
            if self.ready[peer][slot] is None:
                self.ready[peer][slot] = sequence

        def wait():
            self.check(self.ready[rank][slot] == sequence, "notify_generation", (rank, sequence, slot))
            waited.append(self.ready[rank][slot])

        def reset():
            self.check(
                waited == [sequence] and self.ready[rank][slot] == sequence,
                "notify_reset_generation",
                (rank, sequence, slot),
            )
            self.ready[rank][slot] = None

        def read(byte, remote):
            if remote:
                self.check(
                    self.inbox_generations[rank][slot][byte] == sequence,
                    "inbox_generation",
                    (rank, sequence, slot, byte),
                )
                value = self.inboxes[rank][slot][byte]
                self.check(value == packed_input(peer, sequence)[byte], "inbox_byte", (rank, sequence, byte))
                read_peer[byte] = value
                self.consumed[rank][slot][byte] = True
            else:
                self.check(
                    self.input_generations[rank][byte] == sequence, "sum_input_generation", (rank, sequence, byte)
                )
                read_own[byte] = self.input_bytes[rank][byte]

        def sum_lane(lane):
            offset = lane * WORD_BYTES
            own = bytes(read_own[offset + byte] for byte in range(WORD_BYTES))
            other = bytes(read_peer[offset + byte] for byte in range(WORD_BYTES))
            value = struct.unpack("<f", own)[0] + struct.unpack("<f", other)[0]
            self.output[rank][lane] = struct.unpack("<f", struct.pack("<f", value))[0]
            self.output_generations[rank][lane] = sequence

        def consume_lane(lane):
            self.check(self.output_generations[rank][lane] == sequence, "output_generation", (rank, sequence, lane))
            expected = input_values(0, sequence)[lane] + input_values(1, sequence)[lane]
            self.check(self.output[rank][lane] == expected, "output_value", (rank, sequence, lane))
            consumed_output.append(self.output[rank][lane])

        # Shuffle byte order WITHIN a complete producer/copy/read operation.
        # Stream ordering between those operations is never weakened.
        order = list(range(PAYLOAD_BYTES))
        random.Random(sequence * 2 + rank).shuffle(order)
        for byte in order:
            operation("produce", lambda byte=byte: produce(byte))
        for byte in order:
            operation("copy-read", lambda byte=byte: copy_read(byte))
            operation("copy-write", lambda byte=byte: copy_write(byte))
        operation("ready-record", record)
        operation("ready-wait", wait, lambda: self.ready[rank][slot] is not None)
        operation("ready-reset", reset)
        for lane in range(LANES):
            for byte in range(WORD_BYTES):
                index = lane * WORD_BYTES + byte
                operation("sum-read-own", lambda index=index: read(index, False))
                operation("sum-read-inbox", lambda index=index: read(index, True))
            operation("sum-write-output", lambda lane=lane: sum_lane(lane))
        for lane in range(LANES):
            operation("consume", lambda lane=lane: consume_lane(lane))
        operation("observe", lambda: self.observed[rank].append(tuple(consumed_output)))

    def run(self, seed, *, preferred=None):
        rng = random.Random(seed)
        while any(self.queues):
            eligible = [rank for rank in range(2) if self.queues[rank] and self.queues[rank][0][0]()]
            if not eligible:
                self.check(False, "deadlock", [list(queue)[0][2] for queue in self.queues if queue])
                return
            # Long rank-biased bursts model arbitrary arrival/compute skew;
            # no wall-clock sleeps or assumptions about equal device speeds.
            rank = preferred if preferred in eligible else rng.choice(eligible)
            if preferred is None and len(eligible) == 2:
                rank = 0 if rng.random() < (0.97 if seed % 2 else 0.03) else 1
            _, action, label = self.queues[rank].popleft()
            self.trace.append(label)
            action()
            self.steps += 1

    def assert_drained(self):
        assert not any(self.queues)
        assert all(value is None for rank in self.ready for value in rank)
        assert all(consumed for rank in self.consumed for slot in rank for consumed in slot)


@pytest.mark.parametrize("seed", range(400))
def test_two_slots_preserve_every_input_byte_and_output_under_adversarial_skew(seed):
    fabric = ByteFabric()
    windows = (1, 3, 5, 2, 6) if seed % 2 else (7, 1, 4, 1, 6)
    start = 0
    for window, count in enumerate(windows):
        for rank in range(2):
            for sequence in range(start, start + count):
                fabric.enqueue(rank, sequence)
        fabric.run(seed * 17 + window)
        fabric.assert_drained()
        start += count
    expected = [
        tuple(a + b for a, b in zip(input_values(0, sequence), input_values(1, sequence), strict=True))
        for sequence in range(start)
    ]
    assert fabric.observed[0] == fabric.observed[1] == expected
    assert not fabric.violations
    assert start % 2 == 1


@pytest.mark.parametrize("preferred", (0, 1))
def test_maximal_skew_keeps_pingpong_safe_across_odd_bounded_drain_windows(preferred):
    fabric = ByteFabric()
    for base in (0, MAX_PENDING - 1):
        for rank in range(2):
            for sequence in range(base, base + MAX_PENDING - 1):
                fabric.enqueue(rank, sequence)
        fabric.run(0, preferred=preferred)
        fabric.assert_drained()
    assert len(fabric.observed[0]) == 2 * (MAX_PENDING - 1)


@pytest.mark.parametrize("preferred", (0, 1))
def test_single_slot_without_consumed_exposes_byte_overwrite_and_binary_notify_coalescing(preferred):
    fabric = ByteFabric(slots=1, fail_fast=False)
    for rank in range(2):
        for sequence in range(4):
            fabric.enqueue(rank, sequence)
    fabric.run(0, preferred=preferred)
    assert "inbox_overwrite" in fabric.violations
    assert "notify_coalescing" in fabric.violations
    assert "inbox_generation" in fabric.violations
    assert "output_value" in fabric.violations or "deadlock" in fabric.violations


def test_two_reduction_boundaries_and_parity_do_not_reset_after_drains_or_requests():
    policy = PingPongPolicy(rows=32, max_pending=7)
    assert policy.inbox_bytes == 2 * 32 * 4096 * 4
    for sequence in range(93):
        step = policy.step(sequence, BOUNDARIES[sequence % 30], expected_sequence=sequence, pending=sequence % 7)
        assert step.slot == sequence % 2
        assert step.boundary.endswith("attention" if sequence % 2 == 0 else "mlp")
        assert step.boundary.split(":")[0] == str(18 + sequence % 30 // 2)
    assert slot_for_sequence(MAX_SEQUENCE) == 1


@pytest.mark.parametrize("sequence", (True, -1, 0.0, MAX_SEQUENCE + 1))
def test_invalid_sequence_or_wrap_rejected(sequence):
    with pytest.raises(ValueError):
        slot_for_sequence(sequence)


@pytest.mark.parametrize("rows,pending", ((False, 1), (0, 1), (33, 1), (1, 0), (1, MAX_PENDING + 1)))
def test_policy_keeps_existing_geometry_and_submission_budget(rows, pending):
    with pytest.raises(ValueError):
        PingPongPolicy(rows=rows, max_pending=pending)


@pytest.mark.parametrize(
    "sequence,boundary,expected,pending",
    (
        (1, BOUNDARIES[1], 0, 0),
        (0, BOUNDARIES[1], 0, 0),
        (0, BOUNDARIES[0], False, 0),
        (0, BOUNDARIES[0], 0, MAX_PENDING),
        (0, BOUNDARIES[0], 0, True),
    ),
)
def test_out_of_order_omitted_boundary_and_unbounded_enqueue_rejected(sequence, boundary, expected, pending):
    with pytest.raises(ValueError):
        PingPongPolicy().step(sequence, boundary, expected_sequence=expected, pending=pending)
