# SPDX-License-Identifier: Apache-2.0
"""Pure arbitrary-rank dataflow/lifecycle tests; no NPU or HCCL evidence."""

import random
import struct
from collections import Counter, deque
from dataclasses import FrozenInstanceError, replace

import pytest

from tools.sidecard.npu_collective_schedule import (
    MAX_CALL_INDEX,
    MAX_PENDING,
    BinomialSchedule,
    CollectiveQuarantined,
    RankAdmission,
)

LANES = 5


@pytest.mark.parametrize("size", (1, 2, 4, 8, 16, 32, 64, 128))
def test_fixed_recursive_doubling_has_distinct_identity_constant_addresses_and_same_math_steps(size):
    members = tuple(7 + index * 3 for index in reversed(range(size)))
    fixed = BinomialSchedule(members, "float32", LANES, algorithm="recursive-doubling-fixed")
    eager = BinomialSchedule(members, "float32", LANES, algorithm="recursive-doubling")
    assert fixed.graph_safe_per_call is True
    assert fixed.identity != eager.identity
    assert fixed.identity != BinomialSchedule(members, "float32", LANES).identity
    assert fixed.identity == BinomialSchedule(members, "float32", LANES, algorithm="recursive-doubling-fixed").identity
    assert fixed.edges == eager.edges
    for rank in members:
        first = fixed.steps_for_rank(rank, 0)
        for call in (0, 1, 3, 19, MAX_CALL_INDEX):
            assert fixed.steps_for_rank(rank, call) == first
            assert first == tuple(replace(step, sequence=0) for step in eager.steps_for_rank(rank, call))
            assert all(step.sequence == 0 and step.slot == 0 and step.opcode == 0 for step in first)


@pytest.mark.parametrize("size", (3, 5, 6, 7, 9, 65, 127))
def test_fixed_recursive_doubling_never_pads_an_odd_group(size):
    with pytest.raises(ValueError, match="power-of-two"):
        BinomialSchedule(tuple(range(size)), "float16", LANES, algorithm="recursive-doubling-fixed")


def test_fixed_wire_sequence_does_not_reset_monotonic_host_admission():
    schedule = BinomialSchedule((0, 1), "float32", LANES, algorithm="recursive-doubling-fixed")
    owner = RankAdmission(schedule, 0, "a" * 64).begin_prepare().resource_prepared(1).prepare_complete()
    peer = replace(owner, rank=1, prepared_peers=(0,))
    owner = owner.activate([owner.receipt(), peer.receipt()])
    for call in range(9):
        with pytest.raises(ValueError):
            owner.begin_call(call + 1)
        owner = owner.begin_call(call)
        # Repeated graph commands intentionally have the same per-edge fields;
        # begin_call and finish_call retain the separate logical call identity.
        for step in schedule.steps_for_rank(0, call):
            assert step.sequence == 0
            owner = owner.accept_step(step)
        owner = owner.finish_call().complete_window(completed=True)
    assert owner.next_call == 9


def pack(values, dtype):
    return struct.pack("<" + ("f" if dtype == "float32" else "e") * len(values), *values)


def unpack(values, dtype):
    code, width = ("f", 4) if dtype == "float32" else ("e", 2)
    return struct.unpack("<" + code * (len(values) // width), values)


def add(left, right, dtype):
    return pack([a + b for a, b in zip(unpack(left, dtype), unpack(right, dtype), strict=True)], dtype)


def input_vector(position, call, dtype):
    return pack(
        [
            -0.0,
            (position - 8) * 0.125,
            (-1) ** position * (call + 0.25),
            (position % 3 - 1) * 0.0001,
            call - position * 0.03125,
        ],
        dtype,
    )


class FixedHandshakeStreams:
    """Byte-interleaved fixed READY/CONSUMED model on serialized rank streams.

    A sender waits for peer CONSUMED before reusing its peer inbox/READY;
    the peer records CONSUMED only after resetting READY and reading every
    inbox byte. The next peer READY in turn acknowledges the sender's prior
    CONSUMED reset. Notify coalescing and unread-byte overwrites are errors.
    Repeating exactly the same edge operations models address-stable replay,
    not an SDK capture: SDMA visibility/release-acquire semantics are assumed.
    """

    def __init__(self, schedule, calls=7, *, consumed=True):
        self.schedule, self.calls, self.consumed = schedule, calls, consumed
        self.queues = {rank: deque() for rank in schedule.members}
        self.values, self.parts, self.contributors, self.results = {}, {}, {}, {}
        self.inboxes, self.reads, self.notifies = {}, {}, {}
        self.nbytes = LANES * (2 if schedule.dtype == "float16" else 4)
        for rank in schedule.members:
            for peer in schedule.peers(rank):
                self.inboxes[rank, peer] = [None] * self.nbytes
                self.reads[rank, peer] = [None] * self.nbytes
        for rank in schedule.members:
            for call in range(calls):
                self._enqueue(rank, call)

    def _record(self, key, call):
        assert self.notifies.get(key) is None, "binary Notify coalesced before peer reset"
        self.notifies[key] = call

    def _wait(self, key, call):
        assert self.notifies[key] == call, "Notify matched the wrong logical call generation"
        self.notifies[key] = None

    def _enqueue(self, rank, call):
        queue = self.queues[rank]

        def append(action):
            queue.append((lambda: True, action))

        def wait(key):
            queue.append((lambda: self.notifies.get(key) is not None, lambda: self._wait(key, call)))

        def produce():
            self.values[rank] = input_vector(self.schedule.members.index(rank), call, self.schedule.dtype)
            self.contributors[rank] = frozenset({rank})

        append(produce)
        for step in self.schedule.steps_for_rank(rank, call):
            assert step.sequence == step.slot == 0
            outgoing, incoming = (step.peer, rank), (rank, step.peer)
            ready_out, ready_in = ("ready", *outgoing), ("ready", *incoming)
            consumed_out, consumed_in = ("consumed", *outgoing), ("consumed", *incoming)
            scratch = bytearray(self.nbytes)
            for index in range(self.nbytes):

                def write(index=index, outgoing=outgoing):
                    old = self.inboxes[outgoing][index]
                    assert old is None or self.reads[outgoing][index] == old[0], "overwrote unread peer inbox byte"
                    self.inboxes[outgoing][index] = (call, self.values[rank][index])

                append(write)

            def publish(outgoing=outgoing, ready_out=ready_out):
                self.parts[outgoing] = (call, self.contributors[rank])
                self._record(ready_out, call)

            append(publish)
            wait(ready_in)
            for index in range(self.nbytes):

                def read(index=index, incoming=incoming, scratch=scratch):
                    generation, value = self.inboxes[incoming][index]
                    assert generation == call, "inbox contains another call's bytes"
                    assert self.reads[incoming][index] != call, "inbox byte read twice"
                    scratch[index] = value
                    self.reads[incoming][index] = call

                append(read)

            def reduce(incoming=incoming, scratch=scratch, step=step):
                generation, contributors = self.parts[incoming]
                assert generation == call
                assert not self.contributors[rank] & contributors
                assert len(self.contributors[rank]) == len(contributors) == step.stride
                self.values[rank] = add(self.values[rank], bytes(scratch), self.schedule.dtype)
                self.contributors[rank] |= contributors

            append(reduce)
            if self.consumed:
                append(lambda consumed_out=consumed_out: self._record(consumed_out, call))
                wait(consumed_in)

        def consume():
            assert self.contributors[rank] == frozenset(self.schedule.members)
            self.results[rank, call] = self.values[rank]

        append(consume)

    def run(self, seed):
        randomizer = random.Random(seed)
        while any(self.queues.values()):
            eligible = [rank for rank, queue in self.queues.items() if queue and queue[0][0]()]
            assert eligible, "fixed READY/CONSUMED schedule deadlocked"
            rank = min(eligible) if randomizer.random() < 0.8 else randomizer.choice(eligible)
            _, action = self.queues[rank].popleft()
            action()
        assert all(value is None for value in self.notifies.values())
        for rank in self.schedule.members:
            for call in range(self.calls):
                assert self.results[rank, call] == reference_tree(self.schedule, call)


@pytest.mark.parametrize("size", (1, 2, 4, 8))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("seed", range(32))
def test_fixed_handshake_repeated_odd_replays_preserve_bytes_resets_and_complete_vectors(size, dtype, seed):
    schedule = BinomialSchedule(tuple(range(size)), dtype, LANES, algorithm="recursive-doubling-fixed")
    FixedHandshakeStreams(schedule).run(seed)


def test_removing_consumed_exposes_fixed_slot_overwrite_or_notify_reset_race():
    schedule = BinomialSchedule((0, 1), "float32", LANES, algorithm="recursive-doubling-fixed")
    with pytest.raises(AssertionError, match="overwrote|coalesced"):
        FixedHandshakeStreams(schedule, consumed=False).run(0)


def reference_tree(schedule, call):
    values = [input_vector(position, call, schedule.dtype) for position in range(len(schedule.members))]
    stride = 1
    while stride < len(values):
        for position in range(0, len(values), stride * 2):
            if position + stride < len(values):
                values[position] = add(values[position], values[position + stride], schedule.dtype)
        stride *= 2
    return values[0]


class Streams:
    """Independent in-order rank streams with adversarial rendezvous timing."""

    def __init__(self, schedule, calls, *, override_root=None):
        self.schedule, self.calls, self.override_root = schedule, calls, override_root
        self.queues = {rank: deque() for rank in schedule.members}
        self.messages, self.values, self.contributions = {}, {}, {}
        self.retired, self.sequences, self.observed, self.root_results = {}, {}, {}, {}
        self.broadcast_inputs, self.additions = [], 0
        self.trace = []
        for rank in schedule.members:
            self.observed[rank] = []
            for call in range(calls):
                self._enqueue(rank, call)

    def _enqueue(self, rank, call):
        schedule = self.schedule
        queue = self.queues[rank]
        position = schedule.members.index(rank)

        def produce():
            self.values[rank] = input_vector(position, call, schedule.dtype)
            self.contributions[rank] = frozenset({rank})
            self.retired[rank] = False

        queue.append((lambda: True, produce))
        for step in schedule.steps_for_rank(rank, call):
            outgoing = (rank, step.peer, step.edge, step.sequence)
            incoming = (step.peer, rank, step.edge, step.sequence)

            def send(step=step, outgoing=outgoing):
                history = self.sequences.setdefault((rank, step.peer), [])
                assert step.sequence == len(history), "edge phase/sequence skipped or repeated"
                history.append(step.sequence)
                assert outgoing not in self.messages
                if step.phase == "reduce":
                    assert not self.retired[rank], "retired child propagated an extra upward contribution"
                elif step.role == "parent":
                    assert self.contributions[rank] == frozenset(schedule.members)
                if rank == schedule.root and step.phase == "broadcast":
                    if call not in self.root_results:
                        if self.override_root is not None:
                            self.values[rank] = self.override_root
                        self.root_results[call] = self.values[rank]
                self.messages[outgoing] = (
                    step.phase,
                    step.stage,
                    step.role,
                    self.values[rank],
                    self.contributions[rank],
                )
                self.trace.append((rank, call, step.phase, step.peer, step.sequence, step.opcode))

            def receive(step=step, incoming=incoming):
                phase, stage, role, value, members = self.messages.pop(incoming)
                assert phase == step.phase and stage == step.stage
                assert role != step.role
                if step.phase == "reduce":
                    assert step.opcode == 0
                    assert not self.contributions[rank] & members, "duplicate input contribution"
                    if step.role == "parent":
                        self.values[rank] = add(self.values[rank], value, schedule.dtype)
                        self.contributions[rank] |= members
                        self.additions += 1
                    else:
                        # A duplex SUM may also calculate a child result; it is
                        # discarded and cannot escape into a later tree stage.
                        self.retired[rank] = True
                elif step.role == "child":
                    assert step.opcode == 2 and members == frozenset(schedule.members)
                    self.values[rank], self.contributions[rank] = value, members
                    self.broadcast_inputs.append(value)
                else:
                    assert step.opcode == 1
                    # Exact broadcast parent leaves the root bytes unchanged;
                    # it must not execute ADD(root, 0) or read child arithmetic.
                    self.broadcast_inputs.append(self.values[rank])

            queue.append((lambda: True, send))
            queue.append((lambda incoming=incoming: incoming in self.messages, receive))

        def consume():
            assert self.contributions[rank] == frozenset(schedule.members)
            if rank == schedule.root and call not in self.root_results:
                self.root_results[call] = self.values[rank]
            self.observed[rank].append(self.values[rank])

        queue.append((lambda: True, consume))

    def run(self, seed):
        rng = random.Random(seed)
        while any(self.queues.values()):
            eligible = [rank for rank, queue in self.queues.items() if queue and queue[0][0]()]
            assert eligible, "binomial stream schedule deadlocked"
            # Prefer opposite edges of the group in alternating random bursts.
            rank = (min(eligible) if seed % 2 else max(eligible)) if rng.random() < 0.75 else rng.choice(eligible)
            _, action = self.queues[rank].popleft()
            action()
        assert not self.messages


@pytest.mark.parametrize("size", range(1, 66))
def test_every_size_has_exact_tree_edges_two_exchanges_and_matching_peer_steps(size):
    schedule = BinomialSchedule(tuple(range(size)), "float32", 4096)
    assert len(schedule.edges) == size - 1
    assert len(set(schedule.edges)) == size - 1
    for call in (0, 1, 3, 17):
        steps = [step for rank in schedule.members for step in schedule.steps_for_rank(rank, call)]
        assert len(steps) == 4 * (size - 1)
        counts = Counter((step.edge, step.phase) for step in steps)
        assert all(count == 2 for count in counts.values())
        for step in steps:
            matches = [
                peer
                for peer in schedule.steps_for_rank(step.peer, call)
                if peer.peer == step.rank and peer.phase == step.phase
            ]
            assert len(matches) == 1
            peer = matches[0]
            assert (peer.edge, peer.stage, peer.sequence) == (step.edge, step.stage, step.sequence)
            assert peer.role != step.role
            assert step.sequence == 2 * call + (step.phase == "broadcast")
            assert step.slot == (step.phase == "broadcast")


@pytest.mark.parametrize("size", (1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 31, 32, 33, 63, 64, 65))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("seed", range(20))
def test_arbitrary_delays_preserve_full_vectors_and_every_contribution_once(size, dtype, seed):
    members = tuple(7 + 3 * index for index in range(size))
    schedule = BinomialSchedule(members, dtype, LANES)
    simulation = Streams(schedule, 3)
    simulation.run(seed)
    expected = [reference_tree(schedule, call) for call in range(3)]
    for rank in members:
        assert simulation.observed[rank] == expected
    assert simulation.additions == 3 * (size - 1)
    assert all(sequences == list(range(6)) for sequences in simulation.sequences.values())


def test_reverse_broadcast_preserves_exact_signed_zero_and_nan_payload_bits():
    root_bytes = bytes.fromhex("000000804523c17f000000000000807f000080ff")
    schedule = BinomialSchedule(tuple(range(7)), "float32", LANES)
    simulation = Streams(schedule, 2, override_root=root_bytes)
    simulation.run(13)
    assert all(value == root_bytes for values in simulation.observed.values() for value in values)
    assert all(value == root_bytes for value in simulation.broadcast_inputs)
    assert pack([unpack(root_bytes, "float32")[0] + 0.0], "float32") != root_bytes[:4]


def test_configured_order_roots_and_duo_local_siblings_are_explicit():
    schedule = BinomialSchedule((2, 3, 0, 1), "float32", 4)
    assert schedule.root == 2
    assert schedule.steps_for_rank(2)[0].peer == 3
    assert schedule.steps_for_rank(0)[0].peer == 1
    assert schedule.steps_for_rank(2)[1].peer == 0
    assert schedule.steps_for_rank(3)[0].role == "child"
    assert [step.phase for step in schedule.steps_for_rank(3)] == ["reduce", "broadcast"]


def test_identity_is_stable_complete_and_immutable():
    schedule = BinomialSchedule((0, 1, 2, 3), "float32", 4096)
    assert schedule.identity == BinomialSchedule((0, 1, 2, 3), "float32", 4096).identity
    assert len(schedule.identity) == 64
    assert (
        len(
            {
                schedule.identity,
                BinomialSchedule((1, 0, 2, 3), "float32", 4096).identity,
                BinomialSchedule((0, 1, 2, 3), "float16", 4096).identity,
                BinomialSchedule((0, 1, 2, 3), "float32", 8192).identity,
            }
        )
        == 4
    )
    with pytest.raises(FrozenInstanceError):
        schedule.members = (0, 1)


def test_tree_float_order_is_not_an_associativity_or_hccl_equivalence_claim():
    values = [pack([value], "float32") for value in (1e8, 1, -1e8, 1)]
    tree = add(add(values[0], values[1], "float32"), add(values[2], values[3], "float32"), "float32")
    linear = values[0]
    for value in values[1:]:
        linear = add(linear, value, "float32")
    assert unpack(tree, "float32") == (0.0,)
    assert unpack(linear, "float32") == (1.0,)


def active_group(size=4):
    schedule = BinomialSchedule(tuple(range(size)), "float32", 4096)
    prepared = []
    for rank in schedule.members:
        owner = RankAdmission(schedule, rank, "a" * 64).begin_prepare()
        for peer in schedule.peers(rank):
            owner = owner.resource_prepared(peer)
        prepared.append(owner.prepare_complete())
    receipts = tuple(owner.receipt() for owner in prepared)
    return tuple(owner.activate(receipts) for owner in prepared)


def complete_call(owner):
    owner = owner.begin_call(owner.next_call)
    for step in owner.schedule.steps_for_rank(owner.rank, owner.next_call):
        owner = owner.accept_step(step)
    return owner.finish_call()


@pytest.mark.parametrize("size", (1, 3, 4, 5, 7, 65))
def test_all_owned_resources_require_group_drain_and_import_release_before_close(size):
    owners = active_group(size)
    for _ in range(3):
        owners = tuple(complete_call(owner) for owner in owners)
    owners = tuple(owner.complete_window(completed=True) for owner in owners)
    assert all(owner.next_call == 3 and not owner.pending for owner in owners)
    drained = tuple(owner.mark_drained(completed=True) for owner in owners)
    receipts = tuple(owner.receipt() for owner in drained)
    imports_closed = []
    for owner in drained:
        owner = owner.begin_import_release(receipts)
        for peer in owner.schedule.peers(owner.rank):
            owner = owner.import_released(peer)
        imports_closed.append(owner.finish_import_release())
    receipts = tuple(owner.receipt() for owner in imports_closed)
    for owner in imports_closed:
        owner = owner.begin_close(receipts)
        for peer in owner.schedule.peers(owner.rank):
            owner = owner.resource_closed(peer)
        assert owner.finish_close().state == "closed"


def test_partial_failure_retains_every_owned_edge_and_has_no_fallback_transition():
    owner = active_group()[0]
    before = owner.prepared_peers
    owner = owner.begin_call(0).accept_step(owner.schedule.steps_for_rank(0)[0])
    with pytest.raises(ValueError, match="partial"):
        owner.finish_call()
    quarantined = owner.quarantine("second edge enqueue failed after first edge submission")
    assert quarantined.prepared_peers == before
    assert not quarantined.released_peers and not quarantined.closed_peers
    assert quarantined.in_call and quarantined.step_cursor == 1
    for action in (
        lambda: quarantined.begin_call(0),
        lambda: quarantined.mark_drained(completed=True),
        lambda: quarantined.begin_import_release(()),
        lambda: quarantined.finish_close(),
    ):
        with pytest.raises(CollectiveQuarantined):
            action()


def test_partial_setup_or_teardown_cannot_shrink_group_or_free_remaining_edges():
    schedule = BinomialSchedule((0, 1, 2, 3), "float32", 16)
    owner = RankAdmission(schedule, 0, "a" * 64).begin_prepare().resource_prepared(1)
    with pytest.raises(ValueError, match="partially prepared"):
        owner.prepare_complete()
    failed = owner.quarantine("prepare second edge failed")
    assert failed.prepared_peers == (1,)
    owners = tuple(owner.mark_drained(completed=True) for owner in active_group())
    owner = owners[0].begin_import_release(tuple(peer.receipt() for peer in owners)).import_released(1)
    with pytest.raises(ValueError, match="every peer import"):
        owner.finish_import_release()
    failed = owner.quarantine("second imported mapping release uncertain")
    assert failed.released_peers == (1,) and set(failed.prepared_peers) == {1, 2}


@pytest.mark.parametrize(
    "change", ("rank", "members", "bool_member", "bool_peer", "identity", "session", "next_call", "missing")
)
def test_group_receipts_reject_rank_membership_shape_session_or_count_disagreement(change):
    owners = tuple(owner.mark_drained(completed=True) for owner in active_group())
    receipts = [owner.receipt() for owner in owners]
    if change == "rank":
        receipts[1]["rank"] = True
    elif change == "members":
        receipts[1]["members"] = [0, 1]
    elif change == "bool_member":
        receipts[1]["members"][1] = True
    elif change == "bool_peer":
        receipts[0]["peers"][0] = True
    elif change in ("identity", "session"):
        receipts[1][change] = "b" * 64
    elif change == "next_call":
        receipts[1][change] += 1
    else:
        receipts.pop()
    with pytest.raises(ValueError):
        owners[0].begin_import_release(receipts)


def test_stale_step_or_omitted_phase_and_bounded_pending_admission_rejected():
    owner = active_group()[0]
    with pytest.raises(ValueError):
        owner.begin_call(1)
    started = owner.begin_call(0)
    with pytest.raises(ValueError):
        started.accept_step(owner.schedule.steps_for_rank(0)[1])
    with pytest.raises(ValueError):
        started.complete_window(completed=True)
    for _ in range(MAX_PENDING):
        owner = complete_call(owner)
    with pytest.raises(ValueError, match="bounded"):
        owner.begin_call(owner.next_call)
    owner = owner.complete_window(completed=True)
    assert owner.begin_call(MAX_PENDING).next_call == MAX_PENDING
    with pytest.raises(ValueError):
        owner.complete_window(completed=False)


@pytest.mark.parametrize(
    "members,dtype,elements",
    (
        ((), "float32", 1),
        ((0, 0), "float32", 1),
        ((True, 1), "float32", 1),
        ((-1, 0), "float32", 1),
        ([0, 1], "float32", 1),
        ((0,), "bfloat16", 1),
        ((0,), "float32", 0),
        ((0,), "float32", True),
        ((0,), "float32", 1 << 63),
    ),
)
def test_invalid_group_or_fixed_geometry_rejected(members, dtype, elements):
    with pytest.raises(ValueError):
        BinomialSchedule(members, dtype, elements)


def test_rank_membership_and_per_edge_sequence_cannot_wrap_or_reset():
    schedule = BinomialSchedule((2, 3), "float32", 1)
    for rank in (0, True, -1):
        with pytest.raises(ValueError):
            schedule.steps_for_rank(rank)
    for call in (True, -1, MAX_CALL_INDEX + 1):
        with pytest.raises(ValueError):
            schedule.steps_for_rank(2, call)
    assert [step.sequence for step in schedule.steps_for_rank(2, MAX_CALL_INDEX)] == [(1 << 64) - 2, (1 << 64) - 1]
    with pytest.raises(ValueError):
        RankAdmission(schedule, True, "a" * 64)
    with pytest.raises(ValueError, match="every native"):
        RankAdmission(schedule, 2, "a" * 64, state="active")
    wrong = replace(schedule.steps_for_rank(2)[0], sequence=2)
    owner = active_group()[0].begin_call(0)
    with pytest.raises(ValueError):
        owner.accept_step(wrong)
