# SPDX-License-Identifier: Apache-2.0
"""Pure ownership model for two complete-state DMA slots; no device operations.

Transitions accept completed-operation evidence from trusted owning adapters.
They do not enqueue transfers, synchronize devices, or make a blocking bridge
asynchronous. See STAGE_SLOTS.md before binding this model to live allocations.
"""

import math
import threading
from dataclasses import asdict, dataclass, field, fields

from .stage_wire import descriptors

PROTOCOL = "glm-complete-stage-slots-v1"
SCHEMA_VERSION = 1
BRIDGE_UAPI_VERSION = 5
SLOT_COUNT = 2
RANK_COUNT = 2
PHYSICAL_CHIPS = (2, 3)
DEFAULT_CAPACITY = 4 * 1024 * 1024
MAX_ROWS = 32
MAX_CONTEXT = 131072
MAX_DRAIN_SECONDS = 300
MAX_DMA_REGION_BYTES = 8 * 1024 * 1024


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")


def _text(value, name):
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError(f"invalid {name}")


def _same(left, right):
    """Canonical control equality must not equate False with integer zero."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _rank(rank):
    _integer(rank, "rank", maximum=RANK_COUNT - 1)
    return rank


class WouldBlock(RuntimeError):
    """Bounded admission/result retrieval cannot currently make progress."""


class Quarantined(RuntimeError):
    """Allocation ownership must be retained; automatic recovery is forbidden."""


@dataclass(frozen=True)
class Registration:
    """Versioned identity of one session's persistent, private allocations.

    mapping_ids are opaque allocation-set identities, one per slot. The adapter
    must bind each to its CUDA input/result and both chips' HBM/guard handles.
    This model deliberately never owns or serializes those actual handles.
    """

    session: str
    nonce: str
    checkpoint_sha256: str
    mapping_ids: tuple[str, str]
    capacity: int = DEFAULT_CAPACITY
    max_rows: int = MAX_ROWS
    max_context: int = MAX_CONTEXT
    schema_version: int = SCHEMA_VERSION
    bridge_uapi_version: int = BRIDGE_UAPI_VERSION
    protocol: str = PROTOCOL

    def __post_init__(self):
        _text(self.session, "session")
        _text(self.nonce, "nonce")
        if (
            type(self.checkpoint_sha256) is not str
            or len(self.checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
        ):
            raise ValueError("invalid checkpoint digest")
        if type(self.mapping_ids) is not tuple or len(self.mapping_ids) != SLOT_COUNT:
            raise ValueError("exactly two immutable mapping identities required")
        for identity in self.mapping_ids:
            _text(identity, "mapping identity")
        if len(set(self.mapping_ids)) != SLOT_COUNT:
            raise ValueError("slot mapping identities must differ")
        _integer(self.capacity, "slot capacity", minimum=1, maximum=MAX_DMA_REGION_BYTES)
        _integer(self.max_rows, "maximum rows", minimum=1, maximum=MAX_ROWS)
        if type(self.max_context) is not int or self.max_context != MAX_CONTEXT:
            raise ValueError("the full 131072-token context reservation is required")
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
            or type(self.bridge_uapi_version) is not int
            or self.bridge_uapi_version != BRIDGE_UAPI_VERSION
            or self.protocol != PROTOCOL
        ):
            raise ValueError("unqualified registration protocol or bridge UAPI")

    def packet(self, rank):
        value = asdict(self)
        value["mapping_ids"] = list(self.mapping_ids)
        value.update(slots=SLOT_COUNT, rank=_rank(rank), physical_chip=PHYSICAL_CHIPS[rank])
        return value


@dataclass(frozen=True)
class Submission:
    session: str
    nonce: str
    request: str
    request_generation: int
    sequence: int
    slot: int
    generation: int
    mapping_id: str
    first_position: int
    rows: int
    reservation: int
    payload_bytes: int
    mode: str

    def for_rank(self, rank):
        return RankLease(**asdict(self), rank=_rank(rank), physical_chip=PHYSICAL_CHIPS[rank])


@dataclass(frozen=True)
class RankLease(Submission):
    rank: int
    physical_chip: int

    def packet(self):
        return dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, **asdict(self))

    @classmethod
    def from_packet(cls, packet):
        expected = {field.name for field in fields(cls)} | {"schema_version", "protocol"}
        if (
            type(packet) is not dict
            or set(packet) != expected
            or type(packet["schema_version"]) is not int
            or packet["schema_version"] != SCHEMA_VERSION
            or packet["protocol"] != PROTOCOL
        ):
            raise ValueError("invalid lease envelope")
        for name in (
            "request_generation",
            "sequence",
            "slot",
            "generation",
            "first_position",
            "rows",
            "reservation",
            "payload_bytes",
            "rank",
            "physical_chip",
        ):
            _integer(packet[name], name)
        return cls(**{field.name: packet[field.name] for field in fields(cls)})


@dataclass
class _Slot:
    generation: int = 0
    submission: Submission | None = None
    prefix_started: bool = False
    producer_ready: bool = False
    uploaded: set[int] = field(default_factory=set)
    computed: set[int] = field(default_factory=set)
    returned: bool = False
    received: bool = False
    visible: bool = False
    suffix_started: bool = False


class StageSlots:
    """Single-owner, bounded controller for one request and two persistent slots.

    A new request needs CUDA and both-rank reset acknowledgements. Submission
    order, Ascend cache mutation order and CUDA suffix retirement order are
    independent counters. Only prefix B may overlap the complete middle A.
    """

    def __init__(self, registration):
        if not isinstance(registration, Registration):
            raise ValueError("validated Registration required")
        self.registration = registration
        self._owner = threading.get_ident()
        self._slots = [_Slot() for _ in range(SLOT_COUNT)]
        self._offered = set()
        self._registered = set()
        self._unmapped = set()
        self._request = None
        self._request_generation = 0
        self._reservation = None
        self._resets = set()
        self._position = 0
        self._sequence = 0
        self._compute_sequence = 0
        self._receive_sequence = 0
        self._native_busy = [None] * RANK_COUNT
        self._cuda_busy = None
        self._deadline = None
        self._last_poll = None
        self._cancelled = False
        self._quarantine = None
        self._closed = False

    def _check(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("control transitions require the owning thread")
        if self._quarantine is not None:
            raise Quarantined(self._quarantine)
        if self._closed:
            raise RuntimeError("slot session is closed")

    def _slot(self, submission):
        self._check()
        if type(submission) is not Submission:
            raise ValueError("immutable submission required")
        _integer(submission.slot, "slot", maximum=SLOT_COUNT - 1)
        slot = self._slots[submission.slot]
        if slot.submission is None or not _same(asdict(slot.submission), asdict(submission)):
            raise ValueError("stale or altered slot lease")
        return slot

    def _rank_slot(self, lease):
        if type(lease) is not RankLease:
            raise ValueError("physical-rank lease required")
        _rank(lease.rank)
        submission = Submission(**{field.name: getattr(lease, field.name) for field in fields(Submission)})
        slot = self._slot(submission)
        if not _same(asdict(lease), asdict(submission.for_rank(lease.rank))):
            raise ValueError("physical chip or lease identity differs")
        return slot

    def offer_mapping(self, rank):
        """Retain owners before any descriptor can cross to this rank."""
        self._check()
        _rank(rank)
        if self._deadline is not None or rank in self._offered:
            raise ValueError("altered or replayed mapping offer")
        self._offered.add(rank)

    def register(self, rank, packet):
        self._check()
        _rank(rank)
        if (
            self._deadline is not None
            or rank not in self._offered
            or rank in self._registered
            or not _same(packet, self.registration.packet(rank))
        ):
            raise ValueError("altered or replayed slot registration")
        self._registered.add(rank)

    def begin_request(self, request, reservation):
        self._check()
        _text(request, "request")
        _integer(reservation, "request reservation", minimum=1, maximum=self.registration.max_context)
        if self._deadline is not None or self._request is not None or self._registered != set(range(RANK_COUNT)):
            raise RuntimeError("session cannot begin another request")
        self._request = request
        self._request_generation += 1
        self._reservation = reservation
        self._position = 0
        self._resets.clear()
        return self._request_generation

    def _reset_owner(self, request, generation, owner):
        self._check()
        if (
            self._request is None
            or type(generation) is not int
            or (request, generation) != (self._request, self._request_generation)
        ):
            raise ValueError("cache reset belongs to a stale request")
        if type(owner) is not str or owner not in ("cuda", "rank0", "rank1") or owner in self._resets:
            raise ValueError("invalid or replayed cache reset acknowledgement")
        if any(slot.submission is not None for slot in self._slots):
            raise RuntimeError("cannot reset caches while slots are leased")
        return "cuda" if owner == "cuda" else int(owner[-1])

    def reset_started(self, request, generation, owner):
        target = self._reset_owner(request, generation, owner)
        busy = self._cuda_busy if target == "cuda" else self._native_busy[target]
        if busy is not None or self._deadline is not None:
            raise RuntimeError("cache reset overlaps owning runtime work")
        if target == "cuda":
            self._cuda_busy = ("reset", generation)
        else:
            self._native_busy[target] = ("reset", generation)

    def reset_complete(self, request, generation, owner):
        target = self._reset_owner(request, generation, owner)
        busy = self._cuda_busy if target == "cuda" else self._native_busy[target]
        if busy != ("reset", generation):
            raise RuntimeError("cache reset completion lacks its started operation")
        if target == "cuda":
            self._cuda_busy = None
        else:
            self._native_busy[target] = None
        self._resets.add(owner)

    def submit(self, *, first_position, rows, payload_bytes, mode="prefill"):
        """Reserve a free slot without blocking or performing any device work."""
        self._check()
        _integer(first_position, "position")
        _integer(rows, "rows", minimum=1, maximum=self.registration.max_rows)
        _integer(payload_bytes, "payload size", minimum=1, maximum=self.registration.capacity)
        if payload_bytes != descriptors(rows)[1]:
            raise ValueError("payload differs from canonical complete-state geometry")
        if mode not in ("prefill", "decode") or (mode == "decode" and rows != 1):
            raise ValueError("unsupported slot mode")
        if self._deadline is not None or self._request is None or self._resets != {"cuda", "rank0", "rank1"}:
            raise RuntimeError("request is draining, absent, or not reset")
        if first_position != self._position or first_position + rows > self._reservation:
            raise ValueError("reordered cache position or reservation overflow")
        occupied = [slot for slot in self._slots if slot.submission is not None]
        if len(occupied) == SLOT_COUNT or (occupied and (mode == "decode" or occupied[0].submission.mode == "decode")):
            raise WouldBlock("slot credit unavailable; decode remains serialized")
        index = next(i for i, slot in enumerate(self._slots) if slot.submission is None)
        generation = self._slots[index].generation + 1
        submission = Submission(
            self.registration.session,
            self.registration.nonce,
            self._request,
            self._request_generation,
            self._sequence,
            index,
            generation,
            self.registration.mapping_ids[index],
            first_position,
            rows,
            self._reservation,
            payload_bytes,
            mode,
        )
        self._slots[index] = _Slot(generation=generation, submission=submission)
        self._position += rows
        self._sequence += 1
        return submission

    def prefix_started(self, submission):
        slot = self._slot(submission)
        if slot.prefix_started or self._cuda_busy is not None:
            raise RuntimeError("overlapping or replayed CUDA prefix")
        # Prefix cache writes must follow submission order, even when both slots
        # have already been reserved by the bounded caller.
        if any(
            s.submission and s.submission.sequence < submission.sequence and not s.producer_ready for s in self._slots
        ):
            raise RuntimeError("CUDA prefix cache order differs")
        slot.prefix_started = True
        self._cuda_busy = ("prefix", submission.sequence)

    def producer_ready(self, submission):
        slot = self._slot(submission)
        if slot.producer_ready or self._cuda_busy != ("prefix", submission.sequence):
            raise RuntimeError("CUDA producer completion lacks its active prefix")
        slot.producer_ready = True
        self._cuda_busy = None

    def upload_started(self, lease):
        slot = self._rank_slot(lease)
        if not slot.producer_ready or lease.rank in slot.uploaded or self._native_busy[lease.rank] is not None:
            raise RuntimeError("upload lacks producer readiness or overlaps rank work")
        self._native_busy[lease.rank] = ("upload", lease.sequence)

    def upload_complete(self, lease):
        slot = self._rank_slot(lease)
        if self._native_busy[lease.rank] != ("upload", lease.sequence) or lease.rank in slot.uploaded:
            raise RuntimeError("upload completion is stale or has not started")
        self._native_busy[lease.rank] = None
        slot.uploaded.add(lease.rank)

    def compute_started(self, lease):
        slot = self._rank_slot(lease)
        if (
            slot.uploaded != set(range(RANK_COUNT))
            or lease.sequence != self._compute_sequence
            or lease.rank in slot.computed
            or self._native_busy[lease.rank] is not None
        ):
            raise RuntimeError("Ascend cache order, upload quorum, or rank ownership differs")
        self._native_busy[lease.rank] = ("compute", lease.sequence)

    def compute_complete(self, lease):
        slot = self._rank_slot(lease)
        if self._native_busy[lease.rank] != ("compute", lease.sequence) or lease.rank in slot.computed:
            raise RuntimeError("compute completion is stale or has not started")
        self._native_busy[lease.rank] = None
        slot.computed.add(lease.rank)
        if slot.computed == set(range(RANK_COUNT)):
            self._compute_sequence += 1

    def return_started(self, lease):
        slot = self._rank_slot(lease)
        if (
            lease.rank != 0
            or slot.computed != set(range(RANK_COUNT))
            or slot.returned
            or self._native_busy[0] is not None
        ):
            raise RuntimeError("rank0 result return lacks both-rank completion")
        self._native_busy[0] = ("return", lease.sequence)

    def return_complete(self, lease):
        slot = self._rank_slot(lease)
        if lease.rank != 0 or self._native_busy[0] != ("return", lease.sequence) or slot.returned:
            raise RuntimeError("result DMA completion is stale or has not started")
        self._native_busy[0] = None
        slot.returned = True

    def receive(self):
        """Claim the next result in cache order; no waiting, copies or flushes."""
        self._check()
        slot = next((s for s in self._slots if s.submission and s.submission.sequence == self._receive_sequence), None)
        if slot is None or not slot.returned or slot.received:
            raise WouldBlock("ordered result is not available")
        slot.received = True
        return slot.submission

    def cuda_visible(self, submission):
        slot = self._slot(submission)
        if not slot.received or slot.visible:
            raise RuntimeError("visibility flush lacks ordered result receipt")
        slot.visible = True

    def suffix_started(self, submission):
        slot = self._slot(submission)
        if not slot.visible or slot.suffix_started or self._cuda_busy is not None:
            raise RuntimeError("CUDA suffix lacks visibility or conflicts with CUDA prefix/suffix")
        slot.suffix_started = True
        self._cuda_busy = ("suffix", submission.sequence)

    def retire(self, submission):
        """Acknowledge completion of every CUDA result consumer, not its launch."""
        slot = self._slot(submission)
        if self._cuda_busy != ("suffix", submission.sequence) or not slot.suffix_started:
            raise RuntimeError("retirement requires the CUDA suffix consumer completion event")
        self._cuda_busy = None
        self._retire(slot)

    def discard_cancelled(self, submission):
        """Retire an unconsumed result only after a cancelled request is drained."""
        slot = self._slot(submission)
        if not self._cancelled or not slot.visible or slot.suffix_started:
            raise RuntimeError("discard requires cancellation, visibility, and no CUDA consumer")
        self._retire(slot)

    def _retire(self, slot):
        if slot.submission.sequence != self._receive_sequence:
            raise RuntimeError("CUDA suffix cache retirement order differs")
        self._receive_sequence += 1
        slot.submission = None

    def close_request(self):
        self._check()
        if (
            self._request is None
            or any(s.submission is not None for s in self._slots)
            or self._cuda_busy is not None
            or any(value is not None for value in self._native_busy)
        ):
            raise RuntimeError("request still owns slots or is absent")
        self._request = None
        self._reservation = None
        self._resets.clear()

    def accept(self, phase, packet):
        """Validate an exact rank envelope before accepting a worker transition."""
        lease = RankLease.from_packet(packet)
        if phase not in (
            "upload_started",
            "upload_complete",
            "compute_started",
            "compute_complete",
            "return_started",
            "return_complete",
        ):
            raise ValueError("unknown rank transition")
        getattr(self, phase)(lease)

    def begin_drain(self, *, now, timeout, cancelled=False):
        self._check()
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or now < 0
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= MAX_DRAIN_SECONDS
            or type(cancelled) is not bool
        ):
            raise ValueError("invalid bounded drain deadline")
        if self._deadline is not None:
            raise RuntimeError("drain deadline cannot be extended or replayed")
        self._deadline = now + timeout
        self._last_poll = now
        self._cancelled = cancelled

    def poll_drain(self, now):
        self._check()
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or self._deadline is None
            or now < self._last_poll
        ):
            raise ValueError("drain requires monotonic finite timestamps")
        self._last_poll = now
        drained = (
            not any(s.submission is not None for s in self._slots)
            and self._cuda_busy is None
            and all(value is None for value in self._native_busy)
        )
        if now >= self._deadline and (not drained or self._unmapped != self._offered):
            self.quarantine("bounded drain expired before consumer/mapping retirement")
            raise Quarantined(self._quarantine)
        return drained

    def mapping_released(self, rank, packet, *, guard_poison_exact):
        self._check()
        _rank(rank)
        if (
            self._deadline is None
            or rank not in self._offered
            or rank in self._unmapped
            or not _same(packet, self.registration.packet(rank))
            or any(s.submission is not None for s in self._slots)
            or self._cuda_busy is not None
            or any(value is not None for value in self._native_busy)
        ):
            raise RuntimeError("mapping release lacks a drained, matching registered owner")
        if guard_poison_exact is not True:
            self.quarantine("rank lifetime guard differs at mapping release")
            raise Quarantined(self._quarantine)
        self._unmapped.add(rank)

    def resume_after_cancel(self, now):
        """Keep registered allocations only after a completely drained cancel.

        The next request still requires fresh CUDA and both-chip cache resets.
        Once any mapping is released, only shutdown is permitted.
        """
        self._check()
        if not self._cancelled or self._unmapped or not self.poll_drain(now):
            raise RuntimeError("cancel cannot resume before every consumer is retired")
        self.close_request()
        self._deadline = None
        self._last_poll = None
        self._cancelled = False

    def release_owners(self):
        self._check()
        if (
            self._deadline is None
            or self._unmapped != self._offered
            or any(s.submission for s in self._slots)
            or self._cuda_busy is not None
            or any(value is not None for value in self._native_busy)
        ):
            raise RuntimeError("allocation owners must outlive every mapping and consumer")
        self._closed = True

    def quarantine(self, reason):
        if threading.get_ident() != self._owner:
            raise RuntimeError("quarantine transition requires the owning thread")
        _text(reason, "quarantine reason")
        if self._closed:
            raise RuntimeError("closed session cannot be quarantined")
        if self._quarantine is None:
            self._quarantine = reason

    def snapshot(self):
        """Bounded diagnostic receipt, including identities of retained leases."""
        if threading.get_ident() != self._owner:
            raise RuntimeError("snapshot requires the owning thread")
        return dict(
            protocol=PROTOCOL,
            state="quarantined"
            if self._quarantine
            else "closed"
            if self._closed
            else "draining"
            if self._deadline is not None
            else "open",
            reason=self._quarantine,
            offered_ranks=sorted(self._offered),
            registered_ranks=sorted(self._registered),
            unmapped_ranks=sorted(self._unmapped),
            deadline=self._deadline,
            pending=[asdict(s.submission) for s in self._slots if s.submission is not None],
            cuda_busy=self._cuda_busy,
            native_busy=list(self._native_busy),
            next_sequence=self._sequence,
            next_compute_sequence=self._compute_sequence,
            next_receive_sequence=self._receive_sequence,
            payload_host_staging=False,
            owners_releasable=(
                self._deadline is not None
                and self._unmapped == self._offered
                and not self._quarantine
                and not self._closed
                and not any(s.submission for s in self._slots)
                and self._cuda_busy is None
                and all(value is None for value in self._native_busy)
            ),
        )


def memory_requirements(*, capacity=DEFAULT_CAPACITY, cuda_scratch_bytes=0, rank_scratch_bytes=(0, 0)):
    """Additional fixed slot/guard/scratch reservations, excluding model/cache.

    Both chips conservatively reserve result slots. Rank1 packing removal is a
    separate T16 checkpoint and must use a separately authenticated budget.
    """
    _integer(capacity, "slot capacity", minimum=1, maximum=MAX_DMA_REGION_BYTES)
    _integer(cuda_scratch_bytes, "CUDA scratch bytes")
    if type(rank_scratch_bytes) is not tuple or len(rank_scratch_bytes) != RANK_COUNT:
        raise ValueError("one immutable scratch reservation per rank required")
    for value in rank_scratch_bytes:
        _integer(value, "rank scratch bytes")
    per_owner = SLOT_COUNT * 2 * capacity
    return dict(
        slots=SLOT_COUNT,
        capacity_per_slot=capacity,
        cuda_input_result_bytes=per_owner,
        cuda_total_extra_bytes=per_owner + cuda_scratch_bytes,
        rank_hbm_input_result_bytes=[per_owner] * RANK_COUNT,
        rank_hbm_total_extra_bytes=[per_owner + value for value in rank_scratch_bytes],
        rank_host_guard_bytes=[SLOT_COUNT * capacity] * RANK_COUNT,
        host_guard_total_bytes=RANK_COUNT * SLOT_COUNT * capacity,
        excludes="weights, full-context caches, graph pools, runtime reservations, allocator fragmentation",
        hardware_validated=False,
    )
