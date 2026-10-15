# SPDX-License-Identifier: Apache-2.0
"""Immutable binomial/recursive-doubling SUM policies; no device code.

Ordered group members determine the tree and root (members[0]). At stride
1,2,4,... a surviving position divisible by 2*stride merges child position
parent+stride when present; that child becomes inactive upward. Reversing each
rank's upward edges receives the complete root result before forwarding it to
lower-stage children. Thus siblings 0/1 and 2/3 merge before edge 0/2, retaining
Duo locality when the supplied ordered group follows physical Duo placement.

Induction: before an upward stage k, each active rank owns a disjoint contiguous
subtree of at most 2**k members. A parent merges only its adjacent child's
disjoint subtree. Rank zero therefore receives every contribution exactly once.
Reverse traversal delivers that unchanged complete result down every tree edge.
There are N-1 edges and exactly two exchanges per edge per ALLREDUCE: reduction
sequence 2*call (SUM opcode 0), broadcast sequence 2*call+1 (parent opcode 1,
child opcode 2). Broadcast is an EXACT COPY, never an additional zero-add.

Parity invariant: reduction always selects ping-pong slot zero and broadcast
slot one. Each complete ALLREDUCE advances every edge by two, so a captured
whole-ALLREDUCE schedule preserves fixed slot addresses on replay. This is an
address-selection property, NOT graph/Notify/device qualification or permission
to release live graphs/buffers. Native ordering, lifetime and model gates remain.
All steps of one rank require its owning in-order stream: each parent result
must feed the next stage before that stage reads the partial. Each tree edge
retains independent buffers/Notify ownership; this policy does not authorize
cross-stream execution or overlapping communicators on shared scratch.

Opt-in recursive doubling requires a power-of-two group (including one rank).
At stage k every position exchanges a partial SUM with position xor 2**k.
The two partials cover disjoint groups of 2**k contributions, so log2(N) stages
deliver all contributions to every rank without a reverse broadcast. The
supplied member order, rather than numeric global rank, selects XOR partners.
Each edge executes once per ALLREDUCE, sequence=call_index, alternating slots
across calls. Initial product admission is eager only: one captured ALLREDUCE
does NOT preserve slot parity for replay when N>1. A future qualified capture
must contain an even number of SUM calls PER EDGE, plus prove all existing
stream/Notify/lifetime requirements. graph_safe_per_call describes only fixed
slot addresses, not general hardware graph eligibility.

The distinct recursive-doubling-fixed algorithm uses the same XOR SUM tree but
always sequence/slot zero. Each edge must enable the native fixed handshake:
PUSH; record peer.READY; wait/reset own.READY; SUM; record peer.CONSUMED;
wait/reset own.CONSUMED. Before A can overwrite B's inbox or record B.READY
again, A observes B.CONSUMED, recorded after B reset READY and read its inbox.
Before A can record B.CONSUMED again, A must receive B.READY for the next call;
B records that only after resetting its prior CONSUMED. Thus both binary
Notify objects are reset before re-recording, even with arbitrary peer skew.
Fixed addresses allow a per-call graph layout; stream ordering, peer SDMA
visibility, Notify release/acquire semantics, graph/handler lifetime and device
qualification remain required. Host call/admission counters remain monotonic;
the constant wire sequence is not a call identity or permission to overlap.

Floating addition is not associative. This ordered tree can differ from HCCL
for float16/float32, cancellation, overflow and NaN propagation; no bitwise HCCL
equivalence or relaxed model tolerance follows from contribution-set correctness.
The pure admission state below never allocates, submits, synchronizes or frees
resources. The owner must record ANY uncertain partial native failure through
quarantine and retain all remaining resources; no rollback/fallback is provided.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType

PROTOCOL = "sidecard-binomial-sum-v1"
DTYPE_BYTES = MappingProxyType({"float16": 2, "float32": 4})
MAX_PENDING = 64
MAX_CALL_INDEX = ((1 << 64) - 2) // 2
MAX_TENSOR_BYTES = (1 << 63) - 1
STATES = frozenset(
    {
        "new",
        "preparing",
        "prepared",
        "active",
        "drained",
        "imports-releasing",
        "imports-closed",
        "closing",
        "closed",
        "quarantined",
    }
)


def _call_index(value):
    if type(value) is not int or not 0 <= value <= MAX_CALL_INDEX:
        raise ValueError("call index would wrap the per-edge uint64 sequence; start a new drained session")
    return value


@dataclass(frozen=True)
class TreeStep:
    rank: int
    peer: int
    phase: str
    role: str
    stage: int
    stride: int
    edge: tuple
    sequence: int

    def __post_init__(self):
        if (
            any(type(value) is not int or value < 0 for value in (self.rank, self.peer, self.stage, self.sequence))
            or self.rank == self.peer
            or self.phase not in ("reduce", "broadcast", "sum")
            or self.role not in ("parent", "child")
            or type(self.stride) is not int
            or self.stride != 1 << self.stage
            or type(self.edge) is not tuple
            or self.edge != ((self.rank, self.peer) if self.role == "parent" else (self.peer, self.rank))
            or self.sequence > 2 * MAX_CALL_INDEX + 1
            or self.phase != "sum"
            and self.sequence % 2 != (self.phase == "broadcast")
        ):
            raise ValueError("invalid immutable peer/stage/phase/sequence step")

    @property
    def opcode(self):
        return 0 if self.phase in ("reduce", "sum") else (1 if self.role == "parent" else 2)

    @property
    def slot(self):
        return self.sequence & 1


@dataclass(frozen=True)
class BinomialSchedule:
    members: tuple
    dtype: str
    elements: int
    algorithm: str = "binomial"
    _identity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (
            type(self.members) is not tuple
            or not self.members
            or any(type(rank) is not int or rank < 0 for rank in self.members)
            or len(set(self.members)) != len(self.members)
        ):
            raise ValueError("an immutable ordered tuple of unique nonnegative global ranks is required")
        if (
            type(self.dtype) is not str
            or self.dtype not in DTYPE_BYTES
            or type(self.elements) is not int
            or self.elements < 1
        ):
            raise ValueError("positive element count and float16/float32 dtype required")
        if self.elements > MAX_TENSOR_BYTES // DTYPE_BYTES[self.dtype]:
            raise ValueError("fixed tensor byte geometry exceeds the native signed address range")
        if type(self.algorithm) is not str or self.algorithm not in (
            "binomial",
            "recursive-doubling",
            "recursive-doubling-fixed",
        ):
            raise ValueError("algorithm must be exactly binomial, recursive-doubling or recursive-doubling-fixed")
        if self.algorithm in ("recursive-doubling", "recursive-doubling-fixed") and len(self.members) & (
            len(self.members) - 1
        ):
            raise ValueError("recursive-doubling requires a power-of-two group; no padding is permitted")
        object.__setattr__(self, "_identity", self._content_identity())

    @property
    def root(self):
        return self.members[0]

    def _position(self, rank):
        if type(rank) is not int or rank not in self.members:
            raise ValueError("rank is not an exact member of this immutable group")
        return self.members.index(rank)

    def steps_for_rank(self, rank, call_index=0):
        position = self._position(rank)
        call_index = _call_index(call_index)
        if self.algorithm in ("recursive-doubling", "recursive-doubling-fixed"):
            steps = []
            for stage in range(len(self.members).bit_length() - 1):
                stride = 1 << stage
                peer = self.members[position ^ stride]
                parent = not (position & stride)
                steps.append(
                    TreeStep(
                        rank,
                        peer,
                        "sum",
                        "parent" if parent else "child",
                        stage,
                        stride,
                        (rank, peer) if parent else (peer, rank),
                        0 if self.algorithm == "recursive-doubling-fixed" else call_index,
                    )
                )
            return tuple(steps)
        upward = []
        stride, stage = 1, 0
        while stride < len(self.members):
            remainder = position % (2 * stride)
            if remainder == 0 and position + stride < len(self.members):
                peer = self.members[position + stride]
                upward.append((peer, "parent", stage, stride, (rank, peer)))
            elif remainder == stride:
                peer = self.members[position - stride]
                upward.append((peer, "child", stage, stride, (peer, rank)))
                break
            stride *= 2
            stage += 1
        return tuple(
            TreeStep(rank, peer, phase, role, level, width, edge, 2 * call_index + phase_index)
            for phase, phase_index, edges in (("reduce", 0, upward), ("broadcast", 1, reversed(upward)))
            for peer, role, level, width, edge in edges
        )

    def peers(self, rank):
        return tuple(step.peer for step in self.steps_for_rank(rank) if step.phase in ("reduce", "sum"))

    @property
    def graph_safe_per_call(self):
        """Whether one whole call restores per-edge slot addresses, not SDK eligibility."""
        return self.algorithm in ("binomial", "recursive-doubling-fixed") or len(self.members) == 1

    @property
    def edges(self):
        return tuple(
            step.edge
            for rank in self.members
            for step in self.steps_for_rank(rank)
            if step.phase in ("reduce", "sum") and step.role == "parent"
        )

    def _content_identity(self):
        layout = [
            [
                rank,
                [
                    dict(
                        peer=step.peer,
                        phase=step.phase,
                        role=step.role,
                        stage=step.stage,
                        stride=step.stride,
                        edge=step.edge,
                        sequence=step.sequence,
                        opcode=step.opcode,
                    )
                    for step in self.steps_for_rank(rank)
                ],
            ]
            for rank in self.members
        ]
        value = dict(
            protocol=PROTOCOL,
            members=self.members,
            root=self.root,
            dtype=self.dtype,
            elements=self.elements,
            algorithm=self.algorithm,
            schedule=layout,
        )
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def identity(self):
        return self._identity


class CollectiveQuarantined(RuntimeError):
    """No retry/fallback/free is admitted while any owned edge is uncertain."""


@dataclass(frozen=True)
class RankAdmission:
    """Pure immutable state transitions; the backend owns all actual operations.

    Begin preparation/call/release/close BEFORE touching native resources;
    acknowledge each successfully submitted step or released resource AFTER it
    succeeds. On any exception use quarantine(), including partial setup/drain.
    Completion flags are explicit backend attestations, not simulated execution.
    """

    schedule: BinomialSchedule
    rank: int
    session: str
    max_pending: int = MAX_PENDING
    state: str = "new"
    next_call: int = 0
    pending: int = 0
    in_call: bool = False
    step_cursor: int = 0
    prepared_peers: tuple = ()
    released_peers: tuple = ()
    closed_peers: tuple = ()
    failure_reason: str = ""

    def __post_init__(self):
        if not isinstance(self.schedule, BinomialSchedule):
            raise ValueError("immutable binomial schedule required")
        self.schedule._position(self.rank)
        if (
            type(self.session) is not str
            or len(self.session) != 64
            or any(char not in "0123456789abcdef" for char in self.session)
        ):
            raise ValueError("fresh 256-bit shared lowercase hexadecimal session nonce required")
        if type(self.max_pending) is not int or not 1 <= self.max_pending <= MAX_PENDING:
            raise ValueError("bounded host admission window required")
        if self.state not in STATES or type(self.next_call) is not int or not 0 <= self.next_call <= MAX_CALL_INDEX + 1:
            raise ValueError("unknown lifecycle state or invalid monotonic call counter")
        if type(self.pending) is not int or not 0 <= self.pending <= self.max_pending or type(self.in_call) is not bool:
            raise ValueError("invalid outstanding native admission state")
        peers = self.schedule.peers(self.rank)
        for recorded in (self.prepared_peers, self.released_peers, self.closed_peers):
            if (
                type(recorded) is not tuple
                or len(set(recorded)) != len(recorded)
                or any(type(peer) is not int or peer not in peers for peer in recorded)
            ):
                raise ValueError("resource audit includes an unexpected/duplicate peer edge")
        if not set(self.closed_peers) <= set(self.released_peers) <= set(self.prepared_peers):
            raise ValueError("resource release audit omitted its ownership preparation")
        if self.state not in ("new", "preparing", "quarantined") and set(self.prepared_peers) != set(peers):
            raise ValueError("active lifecycle requires every native peer resource")
        if type(self.step_cursor) is not int or self.step_cursor < 0 or not self.in_call and self.step_cursor:
            raise ValueError("invalid partial-call submission cursor")

    def _require(self, *states):
        if self.state == "quarantined":
            raise CollectiveQuarantined("retain all N-rank resources after uncertain partial work")
        if self.state not in states:
            raise RuntimeError(f"collective lifecycle is {self.state}")

    def quarantine(self, reason):
        if self.state == "closed" or not isinstance(reason, str) or not reason:
            raise ValueError("nonempty uncertainty reason and a live resource owner required")
        return replace(self, state="quarantined", failure_reason=reason)

    def begin_prepare(self):
        self._require("new")
        return replace(self, state="preparing")

    def resource_prepared(self, peer):
        self._require("preparing")
        if type(peer) is not int or peer not in self.schedule.peers(self.rank) or peer in self.prepared_peers:
            raise ValueError("native peer resource is unexpected or already prepared")
        return replace(self, prepared_peers=self.prepared_peers + (peer,))

    def prepare_complete(self):
        self._require("preparing")
        if set(self.prepared_peers) != set(self.schedule.peers(self.rank)):
            raise ValueError("cannot activate a partially prepared N-rank resource group")
        return replace(self, state="prepared")

    def receipt(self):
        self._require("prepared", "drained", "imports-closed")
        return dict(
            protocol=PROTOCOL,
            identity=self.schedule.identity,
            session=self.session,
            rank=self.rank,
            phase=self.state,
            members=list(self.schedule.members),
            peers=list(self.schedule.peers(self.rank)),
            next_call=self.next_call,
        )

    def _group(self, receipts, phase):
        values = tuple(receipts)
        if len(values) != len(self.schedule.members):
            raise ValueError("every rank must acknowledge the same complete group")
        seen = set()
        for packet in values:
            rank = packet.get("rank") if isinstance(packet, dict) else None
            if type(rank) is not int or rank not in self.schedule.members or rank in seen:
                raise ValueError("missing, duplicate or foreign rank receipt")
            if any(
                type(packet.get(key)) is not list or any(type(member) is not int for member in packet[key])
                for key in ("members", "peers")
            ):
                raise ValueError("rank membership/peer receipt must retain exact integer types")
            expected = dict(
                protocol=PROTOCOL,
                identity=self.schedule.identity,
                session=self.session,
                rank=rank,
                phase=phase,
                members=list(self.schedule.members),
                peers=list(self.schedule.peers(rank)),
                next_call=self.next_call,
            )
            if set(packet) != set(expected) or any(
                type(packet[key]) is not type(value) or packet[key] != value for key, value in expected.items()
            ):
                raise ValueError("group membership, configuration, phase or monotonic call count differs")
            seen.add(rank)

    def activate(self, prepared):
        self._require("prepared")
        self._group(prepared, "prepared")
        return replace(self, state="active")

    def begin_call(self, call_index):
        self._require("active")
        _call_index(call_index)
        if self.in_call or call_index != self.next_call or self.pending >= self.max_pending:
            raise ValueError("partial/stale call or bounded window requires a completed drain")
        return replace(self, in_call=True, step_cursor=0)

    def accept_step(self, step):
        self._require("active")
        steps = self.schedule.steps_for_rank(self.rank, self.next_call)
        if not self.in_call or self.step_cursor >= len(steps) or step != steps[self.step_cursor]:
            raise ValueError("native step skipped/repeated an edge, phase or peer-local sequence")
        return replace(self, step_cursor=self.step_cursor + 1)

    def finish_call(self):
        self._require("active")
        if not self.in_call or self.step_cursor != len(self.schedule.steps_for_rank(self.rank, self.next_call)):
            raise ValueError("partial N-rank native submission cannot fall back or advance")
        return replace(self, next_call=self.next_call + 1, pending=self.pending + 1, in_call=False, step_cursor=0)

    def complete_window(self, *, completed):
        self._require("active")
        if completed is not True or self.in_call:
            raise ValueError("uncertain local stream completion requires quarantine, not reuse/free")
        return replace(self, pending=0)

    def mark_drained(self, *, completed):
        drained = self.complete_window(completed=completed)
        return replace(drained, state="drained")

    def begin_import_release(self, drained):
        self._require("drained")
        self._group(drained, "drained")
        return replace(self, state="imports-releasing")

    def import_released(self, peer):
        self._require("imports-releasing")
        if type(peer) is not int or peer not in self.prepared_peers or peer in self.released_peers:
            raise ValueError("import release does not belong to a retained peer resource")
        return replace(self, released_peers=self.released_peers + (peer,))

    def finish_import_release(self):
        self._require("imports-releasing")
        if set(self.released_peers) != set(self.prepared_peers):
            raise ValueError("retain exports until every peer import closes")
        return replace(self, state="imports-closed")

    def begin_close(self, imports_closed):
        self._require("imports-closed")
        self._group(imports_closed, "imports-closed")
        return replace(self, state="closing")

    def resource_closed(self, peer):
        self._require("closing")
        if type(peer) is not int or peer not in self.prepared_peers or peer in self.closed_peers:
            raise ValueError("export close does not belong to a retained peer resource")
        return replace(self, closed_peers=self.closed_peers + (peer,))

    def finish_close(self):
        self._require("closing")
        if set(self.closed_peers) != set(self.prepared_peers):
            raise ValueError("not all owned peer exports are closed")
        return replace(self, state="closed")
