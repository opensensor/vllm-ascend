# SPDX-License-Identifier: Apache-2.0
"""Pure recursive-doubling policy proofs; no SDK or device qualification."""

import random
import struct
from dataclasses import FrozenInstanceError, replace

import pytest

from tools.sidecard.npu_collective_schedule import (
    MAX_CALL_INDEX,
    BinomialSchedule,
    RankAdmission,
    TreeStep,
)

SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
LANES = 7
CALLS = 4


def doubling(members, dtype="float32", elements=LANES):
    return BinomialSchedule(members, dtype, elements, algorithm="recursive-doubling")


def rounded(value, dtype):
    code = "e" if dtype == "float16" else "f"
    return struct.unpack("<" + code, struct.pack("<" + code, value))[0]


def balanced_reference(contributions, members, dtype):
    partials = [contributions[rank] for rank in members]
    while len(partials) > 1:
        partials = [
            tuple(rounded(a + b, dtype) for a, b in zip(partials[index], partials[index + 1]))
            for index in range(0, len(partials), 2)
        ]
    return partials[0]


class InterleavedDoubling:
    """A single in-order stream per rank with local two-slot inboxes.

    Payloads are copied before READY; waiting consumes an inbox locally. The
    adversarial scheduler can delay any rank, including across call boundaries.
    This is a protocol/math model, not asynchronous hardware execution evidence.
    """

    def __init__(self, schedule, seed):
        self.schedule, self.random = schedule, random.Random(seed)
        self.inboxes, self.sent, self.received, self.outputs = {}, {}, {}, {}
        self.contributions = {
            call: {
                rank: tuple(rounded(self.random.uniform(-2, 2), schedule.dtype) for _ in range(LANES))
                for rank in schedule.members
            }
            for call in range(CALLS)
        }

    def rank(self, rank):
        for call in range(CALLS):
            partial, contributors = self.contributions[call][rank], frozenset({rank})
            for step in self.schedule.steps_for_rank(rank, call):
                outgoing = (step.peer, rank, step.slot)
                assert outgoing not in self.inboxes, "overwrote unread peer inbox"
                history = self.sent.setdefault((rank, step.peer), [])
                assert step.sequence == len(history)
                history.append((step.phase, step.stage, step.sequence))
                self.inboxes[outgoing] = (step, partial, contributors)
                yield "published"
                incoming = (rank, step.peer, step.slot)
                while incoming not in self.inboxes:
                    yield "waiting"
                peer, peer_partial, peer_contributors = self.inboxes.pop(incoming)
                assert (peer.phase, peer.stage, peer.sequence) == (step.phase, step.stage, step.sequence)
                assert peer.role != step.role and peer.edge == step.edge
                assert not contributors.intersection(peer_contributors)
                assert len(contributors) == len(peer_contributors) == step.stride
                contributors |= peer_contributors
                partial = tuple(rounded(a + b, self.schedule.dtype) for a, b in zip(partial, peer_partial))
                self.received.setdefault((rank, step.peer), []).append(step.sequence)
                yield "added"
            assert contributors == frozenset(self.schedule.members)
            self.outputs[rank, call] = partial
            yield "completed"

    def run(self):
        pending = {rank: self.rank(rank) for rank in self.schedule.members}
        # Alternating random bursts let a rank run ahead until a true peer wait.
        limit = CALLS * max(1, len(pending)) * 1000
        ticks = 0
        while pending:
            rank = self.random.choice(tuple(pending))
            for _ in range(self.random.randint(1, 11)):
                ticks += 1
                assert ticks < limit, "recursive-doubling model stalled"
                try:
                    next(pending[rank])
                except StopIteration:
                    del pending[rank]
                    break
        assert not self.inboxes
        for rank in self.schedule.members:
            for call in range(CALLS):
                assert self.outputs[rank, call] == balanced_reference(
                    self.contributions[call], self.schedule.members, self.schedule.dtype
                )
            for peer in self.schedule.peers(rank):
                assert self.received[rank, peer] == list(range(CALLS))
                assert len(self.sent[rank, peer]) == CALLS


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("seed", (19, 821))
@pytest.mark.parametrize("permuted", (False, True))
def test_random_interleavings_contributions_vector_math_and_slot_reuse(size, dtype, seed, permuted):
    members = tuple(index * 11 + 7 for index in range(size))
    if permuted:
        members = tuple(random.Random(seed + size).sample(members, size))
    model = InterleavedDoubling(doubling(members, dtype), seed)
    model.run()


@pytest.mark.parametrize("size", SIZES)
def test_exact_xor_peers_stages_roles_and_matching_sum_phases(size):
    members = tuple(reversed(tuple(index * 5 + 2 for index in range(size))))
    schedule = doubling(members)
    stages = size.bit_length() - 1
    assert len(schedule.edges) == size * stages // 2
    assert len(set(schedule.edges)) == len(schedule.edges)
    for call in (0, 1, 2, 3, MAX_CALL_INDEX):
        for position, rank in enumerate(members):
            steps = schedule.steps_for_rank(rank, call)
            assert len(steps) == stages
            assert schedule.peers(rank) == tuple(members[position ^ (1 << k)] for k in range(stages))
            assert len(set(schedule.peers(rank))) == stages
            for stage, step in enumerate(steps):
                peer_position = position ^ (1 << stage)
                assert step.peer == members[peer_position]
                assert step.stage == stage and step.stride == 1 << stage
                assert step.phase == "sum" and step.opcode == 0
                assert step.sequence == call and step.slot == call % 2
                assert (step.role == "parent") == (position & (1 << stage) == 0)
                peer = schedule.steps_for_rank(step.peer, call)[stage]
                assert peer.peer == rank and peer.role != step.role
                assert (peer.edge, peer.sequence, peer.phase, peer.slot) == (
                    step.edge,
                    step.sequence,
                    step.phase,
                    step.slot,
                )


def test_nonmonotonic_order_uses_positions_and_preserves_duo_first_stage():
    schedule = doubling((7, 3, 9, 2))
    assert schedule.root == 7
    assert schedule.peers(7) == (3, 9)
    assert schedule.peers(3) == (7, 2)
    assert schedule.peers(9) == (2, 7)
    assert schedule.peers(2) == (9, 3)
    assert schedule.edges == ((7, 3), (7, 9), (3, 2), (9, 2))


@pytest.mark.parametrize("size", SIZES)
def test_graph_address_eligibility_requires_even_rd_calls_per_edge(size):
    schedule = doubling(tuple(range(size)))
    assert schedule.graph_safe_per_call is (size == 1)
    assert BinomialSchedule(tuple(range(size)), "float32", LANES).graph_safe_per_call is True
    for rank in schedule.members:
        first = schedule.steps_for_rank(rank, 0)
        odd = schedule.steps_for_rank(rank, 1)
        even = schedule.steps_for_rank(rank, 2)
        assert [(s.peer, s.slot) for s in first] == [(s.peer, s.slot) for s in even]
        assert all(a.slot != b.slot for a, b in zip(first, odd))


def test_algorithm_identity_covers_mode_order_dtype_shape_and_is_frozen():
    schedule = doubling((7, 3, 9, 2))
    assert schedule.identity == doubling((7, 3, 9, 2)).identity
    identities = {
        schedule.identity,
        doubling((3, 7, 9, 2)).identity,
        doubling((7, 3, 9, 2), "float16").identity,
        doubling((7, 3, 9, 2), elements=LANES + 1).identity,
        BinomialSchedule((7, 3, 9, 2), "float32", LANES).identity,
    }
    assert len(identities) == 5 and all(len(value) == 64 for value in identities)
    assert doubling((7,)).identity != BinomialSchedule((7,), "float32", LANES).identity
    with pytest.raises(FrozenInstanceError):
        schedule.algorithm = "binomial"


@pytest.mark.parametrize("members", ((0,), (0, 1), (0, 1, 2), (7, 3, 9, 2)))
def test_explicit_binomial_retains_default_steps_and_two_exchange_sequences(members):
    implicit = BinomialSchedule(members, "float32", LANES)
    explicit = BinomialSchedule(members, "float32", LANES, algorithm="binomial")
    assert implicit.identity == explicit.identity and implicit.edges == explicit.edges
    for rank in members:
        for call in (0, 1, MAX_CALL_INDEX):
            assert implicit.steps_for_rank(rank, call) == explicit.steps_for_rank(rank, call)
            for step in implicit.steps_for_rank(rank, call):
                assert step.phase in ("reduce", "broadcast")
                assert step.sequence == 2 * call + (step.phase == "broadcast")
                assert step.slot == (step.phase == "broadcast")


@pytest.mark.parametrize("size", (3, 5, 6, 7, 9, 10, 12, 127, 129))
def test_non_power_of_two_rd_is_rejected_without_padding_or_fallback(size):
    with pytest.raises(ValueError, match="power-of-two"):
        doubling(tuple(range(size)))
    assert BinomialSchedule(tuple(range(size)), "float32", LANES).algorithm == "binomial"


@pytest.mark.parametrize("algorithm", (None, True, 0, "recursive_doubling", "ring", "Recursive-Doubling", [], {}))
def test_algorithm_selection_rejects_unknown_values_and_coercions(algorithm):
    with pytest.raises(ValueError, match="algorithm"):
        BinomialSchedule((0, 1), "float32", LANES, algorithm=algorithm)


@pytest.mark.parametrize("members", ((), (0, 0), (False, 1), (0, True), (0, -1), (0, "1"), [0, 1]))
def test_invalid_membership_is_rejected_before_schedule_generation(members):
    with pytest.raises(ValueError):
        doubling(members)


@pytest.mark.parametrize("call", (-1, True, 1.0, "0", MAX_CALL_INDEX + 1))
def test_invalid_call_indices_fail_closed(call):
    with pytest.raises(ValueError, match="call index"):
        doubling((0, 1)).steps_for_rank(0, call)


@pytest.mark.parametrize("sequence", (0, 1, 2, 3, (1 << 64) - 1))
def test_sum_steps_accept_either_slot_without_relaxing_binomial_phase_parity(sequence):
    step = TreeStep(7, 2, "sum", "parent", 0, 1, (7, 2), sequence)
    assert step.opcode == 0 and step.slot == sequence % 2
    with pytest.raises(ValueError):
        replace(step, phase="reduce" if sequence % 2 else "broadcast")
    with pytest.raises(ValueError):
        replace(step, role="child")


def test_rank_admission_accepts_rd_and_rejects_reordered_or_stale_steps():
    schedule = doubling((7, 3, 9, 2))
    owner = RankAdmission(schedule, 7, "c" * 64).begin_prepare()
    for peer in schedule.peers(7):
        owner = owner.resource_prepared(peer)
    owner = owner.prepare_complete()
    prepared_receipts = [
        replace(owner, rank=rank, prepared_peers=schedule.peers(rank)).receipt() for rank in schedule.members
    ]
    owner = owner.activate(prepared_receipts).begin_call(0)
    steps = schedule.steps_for_rank(7)
    with pytest.raises(ValueError, match="skipped/repeated"):
        owner.accept_step(steps[1])
    owner = owner.accept_step(steps[0])
    with pytest.raises(ValueError, match="skipped/repeated"):
        owner.accept_step(steps[0])
    with pytest.raises(ValueError, match="partial"):
        owner.finish_call()
    owner = owner.accept_step(steps[1]).finish_call().complete_window(completed=True).begin_call(1)
    with pytest.raises(ValueError, match="skipped/repeated"):
        owner.accept_step(steps[0])
    for step in schedule.steps_for_rank(7, 1):
        owner = owner.accept_step(step)
    assert owner.finish_call().next_call == 2
