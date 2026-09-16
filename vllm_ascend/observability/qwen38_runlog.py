# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run-log and first-fatal-rank observability for the Qwen4Exp 1M 310P path (plan TOBS, R13).

Every server run must leave behind a single, self-describing artifact that lets a
reader reconstruct exactly what happened: the pinned software revisions (plan
T0.2), the hardware topology and this rank's identity (plan T0.3), token limits,
per-rank component memory (plan T0.5), host PLE/pinned/RSS/swap/NUMA footprint,
long-prefill chunk progress and throughput, PLE/QSA transfer+hit metrics (plan
T4.4 schema), hybrid-state lifecycle events, per-collective link-class tracing
(R3), and -- most importantly on a crash -- the *first* fatal error together with
the rank it happened on and the full traceback. The run log never degrades a
failure to a bare "worker died".

This module is a pure-Python *aggregator*: it composes sub-reports produced by
T0.2/T0.3/T0.5/T4.4 (each of which already emits a ``to_dict``) and adds the
collective trace and first-fatal capture on top. It imports no ``torch_npu`` and
touches no accelerator at load, so the whole thing is host-testable with synthetic
inputs. Sub-reports may be passed either as their dataclass objects (anything with
a ``to_dict``) or as already-serialized dicts.

Output: a machine-readable JSON document plus a human-readable log, written
side-by-side by :meth:`RunLog.write`.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1

_BYTES_PER_GIB = 1024**3

# Link locality classes, mirroring tools/qwen38_1m/hw_probe.py so a collective
# annotation lines up with the probe's topology classification (R3).
LINK_WITHIN_CARD = "within_card"
LINK_CROSS_CARD = "cross_card"
LINK_UNKNOWN = "unknown"

# The five run-log PLE/QSA metric fields that MUST be present for a transfer/hit
# section to be considered complete. This mirrors the leading, required entries of
# ``PLE_PREFETCH_METRIC_FIELDS`` (plan T4.4) but is duplicated here as a plain
# tuple so importing the run log never drags in the torch-dependent prefetch
# module. :func:`ple_metric_schema` returns the authoritative T4.4 tuple lazily.
REQUIRED_TRANSFER_METRIC_FIELDS: tuple[str, ...] = (
    "host_bytes",
    "page_faults",
    "transfer_bytes",
    "hit_rate",
    "lookup_latency",
)


def ple_metric_schema() -> tuple[str, ...]:
    """Return the authoritative T4.4 metric field tuple (imported lazily).

    Kept lazy so the run log stays import-light and host-safe: the T4.4 prefetch
    module imports ``torch``, which callers of a pure logging aggregator should
    not be forced to pay for.
    """
    from vllm_ascend.models.qwen4_exp.ple_prefetch import PLE_PREFETCH_METRIC_FIELDS

    return PLE_PREFETCH_METRIC_FIELDS


def _as_dict(report: Any) -> dict | None:
    """Coerce a sub-report (object with ``to_dict`` or a plain dict) to a dict."""
    if report is None:
        return None
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(report, dict):
        return dict(report)
    raise TypeError(f"expected a dict or an object with to_dict(), got {type(report).__name__}")


def classify_link(topology: dict | None, src_chip: int, dst_chip: int) -> str:
    """Classify a chip->chip collective link as within/cross-card from a topology.

    ``topology`` is a T0.3 :meth:`ProbeReport.to_dict` document. Locality is
    derived from each chip's ``card_id`` (two 310P chips on one 300I Duo card share
    the faster within-card link), matching ``hw_probe.classify_links``. Returns
    :data:`LINK_UNKNOWN` when either chip is absent from the topology (so a partial
    probe never silently mislabels traffic).
    """
    if topology is None:
        return LINK_UNKNOWN
    chip_card = {c["chip_id"]: c["card_id"] for c in topology.get("chips", [])}
    src_card = chip_card.get(src_chip)
    dst_card = chip_card.get(dst_chip)
    if src_card is None or dst_card is None:
        return LINK_UNKNOWN
    return LINK_WITHIN_CARD if src_card == dst_card else LINK_CROSS_CARD


@dataclass
class RankIdentity:
    """Which rank this run log belongs to and the chip/card it runs on."""

    rank: int
    world_size: int
    local_rank: int | None = None
    chip_id: int | None = None
    card_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "chip_id": self.chip_id,
            "card_id": self.card_id,
        }


@dataclass
class HostMemorySnapshot:
    """Host-side (not device) memory footprint observed for a rank."""

    rank: int
    ple_bytes: int | None = None
    pinned_bytes: int | None = None
    rss_bytes: int | None = None
    swap_bytes: int | None = None
    numa_node: int | None = None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "ple_bytes": self.ple_bytes,
            "pinned_bytes": self.pinned_bytes,
            "rss_bytes": self.rss_bytes,
            "swap_bytes": self.swap_bytes,
            "numa_node": self.numa_node,
        }


@dataclass
class CollectiveTraceEntry:
    """One HCCL collective, annotated with its link locality class (R3).

    ``link_class`` is one of :data:`LINK_WITHIN_CARD` / :data:`LINK_CROSS_CARD` /
    :data:`LINK_UNKNOWN`, derived from the supplied topology so D4 can compare
    Candidate A vs B collective cost by link.
    """

    op: str
    src_chip: int
    dst_chip: int
    link_class: str
    num_bytes: int
    duration_s: float | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "src_chip": self.src_chip,
            "dst_chip": self.dst_chip,
            "link_class": self.link_class,
            "num_bytes": self.num_bytes,
            "duration_s": self.duration_s,
            "timestamp": self.timestamp,
        }


@dataclass
class LifecycleEvent:
    """A hybrid-state lifecycle event (e.g. GDN state alloc/reset/free)."""

    event: str
    phase: str
    rank: int | None = None
    detail: dict | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "phase": self.phase,
            "rank": self.rank,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class ChunkProgress:
    """Progress of one completed prefill/decode chunk on a long (1M) run."""

    chunk_index: int
    num_chunks: int | None
    tokens_done: int
    tokens_total: int | None
    elapsed_s: float
    tokens_per_s: float | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "num_chunks": self.num_chunks,
            "tokens_done": self.tokens_done,
            "tokens_total": self.tokens_total,
            "elapsed_s": self.elapsed_s,
            "tokens_per_s": self.tokens_per_s,
            "timestamp": self.timestamp,
        }


@dataclass
class Heartbeat:
    """A long-prefill progress heartbeat ping (in-flight, not chunk completion)."""

    tokens_done: int
    tokens_total: int | None
    elapsed_s: float
    tokens_per_s: float | None = None
    note: str | None = None
    timestamp: float | None = None

    @property
    def fraction(self) -> float | None:
        if not self.tokens_total:
            return None
        return self.tokens_done / self.tokens_total

    def to_dict(self) -> dict:
        return {
            "tokens_done": self.tokens_done,
            "tokens_total": self.tokens_total,
            "elapsed_s": self.elapsed_s,
            "tokens_per_s": self.tokens_per_s,
            "fraction": self.fraction,
            "note": self.note,
            "timestamp": self.timestamp,
        }


@dataclass
class FirstFatal:
    """The first fatal error captured, with rank and full traceback preserved."""

    rank: int
    exception_type: str
    message: str
    traceback: str
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback": self.traceback,
            "timestamp": self.timestamp,
        }


class RunLog:
    """Aggregates every run-log section for a Qwen4Exp 1M 310P run (plan TOBS).

    A single instance is created per server run (per rank). Sub-reports from
    T0.2/T0.3/T0.5/T4.4 are attached via the ``set_*`` methods; live events
    (collectives, lifecycle, chunk progress, heartbeats, fatals) are appended as
    they occur. :meth:`to_dict` / :meth:`to_json` render the machine artifact and
    :meth:`human_summary` the human log; :meth:`write` emits both side-by-side.

    The **first** fatal error wins: :meth:`record_fatal` never overwrites an
    earlier capture, so the root cause -- and the rank it happened on, with the
    traceback -- survives any cascade of secondary failures.
    """

    def __init__(
        self,
        *,
        run_id: str,
        world_size: int,
        rank: int | None = None,
        now=time.time,
    ) -> None:
        self.run_id = run_id
        self.world_size = world_size
        self._now = now
        self.started_at = now()
        self.rank_identity: RankIdentity | None = None
        if rank is not None:
            self.rank_identity = RankIdentity(rank=rank, world_size=world_size)
        self.token_limits: dict | None = None
        self._revisions: dict | None = None
        self._topology: dict | None = None
        self._memory: dict | None = None
        self._ple_metrics: dict | None = None
        self._qsa_metrics: dict | None = None
        self.host_memory: list[HostMemorySnapshot] = []
        self.collective_trace: list[CollectiveTraceEntry] = []
        self.lifecycle_events: list[LifecycleEvent] = []
        self.chunk_progress: list[ChunkProgress] = []
        self.heartbeats: list[Heartbeat] = []
        self._first_fatal: FirstFatal | None = None
        self.fatal_count = 0

    # -- sub-report attachment -------------------------------------------- #

    def set_rank_identity(self, identity: RankIdentity) -> None:
        self.rank_identity = identity

    def set_revisions(self, report: Any) -> None:
        """Attach the T0.2 environment freeze (revisions) sub-report."""
        self._revisions = _as_dict(report)

    def set_topology(self, report: Any) -> None:
        """Attach the T0.3 hardware/topology probe sub-report.

        The topology is also used to classify collective link locality (R3).
        """
        self._topology = _as_dict(report)

    def set_memory(self, report: Any) -> None:
        """Attach the T0.5 per-rank component memory accounting sub-report."""
        self._memory = _as_dict(report)

    def set_token_limits(
        self,
        *,
        max_model_len: int,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        prefill_chunk_size: int | None = None,
    ) -> None:
        self.token_limits = {
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "prefill_chunk_size": prefill_chunk_size,
        }

    def set_ple_metrics(self, metrics: Any) -> None:
        """Attach PLE prefetch transfer/hit metrics (T4.4 schema)."""
        self._ple_metrics = _as_dict(metrics)

    def set_qsa_metrics(self, metrics: Any) -> None:
        """Attach QSA transfer/hit metrics (same T4.4 metric schema)."""
        self._qsa_metrics = _as_dict(metrics)

    def record_host_memory(
        self,
        rank: int,
        *,
        ple_bytes: int | None = None,
        pinned_bytes: int | None = None,
        rss_bytes: int | None = None,
        swap_bytes: int | None = None,
        numa_node: int | None = None,
    ) -> HostMemorySnapshot:
        snapshot = HostMemorySnapshot(
            rank=rank,
            ple_bytes=ple_bytes,
            pinned_bytes=pinned_bytes,
            rss_bytes=rss_bytes,
            swap_bytes=swap_bytes,
            numa_node=numa_node,
        )
        self.host_memory.append(snapshot)
        return snapshot

    # -- live events ------------------------------------------------------- #

    def record_collective(
        self,
        op: str,
        *,
        src_chip: int,
        dst_chip: int,
        num_bytes: int,
        duration_s: float | None = None,
        link_class: str | None = None,
    ) -> CollectiveTraceEntry:
        """Record one HCCL collective, annotated with its link locality class.

        When ``link_class`` is not supplied it is derived from the attached T0.3
        topology (:func:`classify_link`), so every trace entry carries a
        within/cross-card annotation even if the caller does not classify it.
        """
        if link_class is None:
            link_class = classify_link(self._topology, src_chip, dst_chip)
        entry = CollectiveTraceEntry(
            op=op,
            src_chip=src_chip,
            dst_chip=dst_chip,
            link_class=link_class,
            num_bytes=num_bytes,
            duration_s=duration_s,
            timestamp=self._now(),
        )
        self.collective_trace.append(entry)
        return entry

    def record_lifecycle(
        self,
        event: str,
        *,
        phase: str,
        rank: int | None = None,
        detail: dict | None = None,
    ) -> LifecycleEvent:
        """Record a hybrid-state lifecycle event (e.g. GDN state alloc/reset/free)."""
        lifecycle = LifecycleEvent(
            event=event,
            phase=phase,
            rank=rank,
            detail=detail,
            timestamp=self._now(),
        )
        self.lifecycle_events.append(lifecycle)
        return lifecycle

    def record_chunk_progress(
        self,
        *,
        chunk_index: int,
        tokens_done: int,
        elapsed_s: float,
        num_chunks: int | None = None,
        tokens_total: int | None = None,
    ) -> ChunkProgress:
        """Record completion of a prefill/decode chunk with a tokens/s figure."""
        tokens_per_s = (tokens_done / elapsed_s) if elapsed_s > 0 else None
        progress = ChunkProgress(
            chunk_index=chunk_index,
            num_chunks=num_chunks,
            tokens_done=tokens_done,
            tokens_total=tokens_total,
            elapsed_s=elapsed_s,
            tokens_per_s=tokens_per_s,
            timestamp=self._now(),
        )
        self.chunk_progress.append(progress)
        return progress

    def heartbeat(
        self,
        *,
        tokens_done: int,
        elapsed_s: float,
        tokens_total: int | None = None,
        note: str | None = None,
    ) -> Heartbeat:
        """Emit a long-prefill progress heartbeat (in-flight ping for 1M runs)."""
        tokens_per_s = (tokens_done / elapsed_s) if elapsed_s > 0 else None
        ping = Heartbeat(
            tokens_done=tokens_done,
            tokens_total=tokens_total,
            elapsed_s=elapsed_s,
            tokens_per_s=tokens_per_s,
            note=note,
            timestamp=self._now(),
        )
        self.heartbeats.append(ping)
        return ping

    def record_fatal(self, rank: int, exc: BaseException) -> FirstFatal:
        """Capture a fatal error; the FIRST one wins and preserves its traceback.

        The full traceback is formatted from the exception's ``__traceback__`` so
        the artifact never degrades a crash to a bare "worker died". Subsequent
        fatals increment :attr:`fatal_count` but never overwrite the root cause.
        """
        self.fatal_count += 1
        if self._first_fatal is not None:
            return self._first_fatal
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._first_fatal = FirstFatal(
            rank=rank,
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback=tb,
            timestamp=self._now(),
        )
        return self._first_fatal

    @property
    def first_fatal(self) -> FirstFatal | None:
        return self._first_fatal

    # -- derived views ----------------------------------------------------- #

    def collective_summary(self) -> dict:
        """Aggregate collective bytes/count by link class (for D4 A/B comparison)."""
        summary: dict[str, dict[str, int]] = {}
        for entry in self.collective_trace:
            bucket = summary.setdefault(entry.link_class, {"count": 0, "bytes": 0})
            bucket["count"] += 1
            bucket["bytes"] += entry.num_bytes
        return summary

    def missing_sections(self) -> list[str]:
        """Names of the required run-log sections that are still unpopulated."""
        required = {
            "revisions": self._revisions,
            "topology": self._topology,
            "token_limits": self.token_limits,
            "memory": self._memory,
            "ple_metrics": self._ple_metrics,
            "qsa_metrics": self._qsa_metrics,
            "rank_identity": self.rank_identity,
        }
        missing = [name for name, value in required.items() if not value]
        if not self.lifecycle_events:
            missing.append("lifecycle_events")
        return missing

    # -- serialization ----------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "world_size": self.world_size,
            "started_at": self.started_at,
            "rank_identity": self.rank_identity.to_dict() if self.rank_identity else None,
            "token_limits": self.token_limits,
            "revisions": self._revisions,
            "topology": self._topology,
            "memory": self._memory,
            "host_memory": [h.to_dict() for h in self.host_memory],
            "ple_metrics": self._ple_metrics,
            "qsa_metrics": self._qsa_metrics,
            "collective_trace": [c.to_dict() for c in self.collective_trace],
            "collective_summary": self.collective_summary(),
            "lifecycle_events": [e.to_dict() for e in self.lifecycle_events],
            "chunk_progress": [p.to_dict() for p in self.chunk_progress],
            "heartbeats": [h.to_dict() for h in self.heartbeats],
            "first_fatal": self._first_fatal.to_dict() if self._first_fatal else None,
            "fatal_count": self.fatal_count,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def human_summary(self) -> str:
        lines = [
            f"Qwen4Exp 310P run log (run_id={self.run_id}, world_size={self.world_size}, schema v{SCHEMA_VERSION})"
        ]
        if self.rank_identity is not None:
            ident = self.rank_identity
            lines.append(f"  rank {ident.rank}/{ident.world_size} (chip {ident.chip_id}, card {ident.card_id})")
        if self.token_limits is not None:
            lines.append(f"  token limits: {self.token_limits}")
        lines.append(f"  revisions: {'present' if self._revisions else 'MISSING'}")
        lines.append(f"  topology: {'present' if self._topology else 'MISSING'}")
        lines.append(f"  per-rank memory: {'present' if self._memory else 'MISSING'}")
        lines.append(f"  PLE metrics: {'present' if self._ple_metrics else 'MISSING'}")
        lines.append(f"  QSA metrics: {'present' if self._qsa_metrics else 'MISSING'}")
        summary = self.collective_summary()
        if summary:
            parts = ", ".join(
                f"{cls}: {stats['count']} ops / {stats['bytes'] / _BYTES_PER_GIB:.2f} GiB"
                for cls, stats in sorted(summary.items())
            )
            lines.append(f"  collectives by link: {parts}")
        lines.append(f"  lifecycle events: {len(self.lifecycle_events)}")
        lines.append(f"  chunk progress records: {len(self.chunk_progress)}")
        lines.append(f"  heartbeats: {len(self.heartbeats)}")
        if self.chunk_progress:
            last = self.chunk_progress[-1]
            tps = f"{last.tokens_per_s:.1f}" if last.tokens_per_s is not None else "n/a"
            lines.append(f"  last chunk: #{last.chunk_index} tokens/s={tps}")
        missing = self.missing_sections()
        if missing:
            lines.append(f"  MISSING sections: {missing}")
        if self._first_fatal is not None:
            fatal = self._first_fatal
            lines.append(
                f"  FIRST FATAL on rank {fatal.rank}: "
                f"{fatal.exception_type}: {fatal.message} "
                f"(total fatals: {self.fatal_count})"
            )
            lines.append("  --- traceback (preserved) ---")
            lines.extend(f"  {ln}" for ln in fatal.traceback.rstrip().splitlines())
        else:
            lines.append("  no fatal error recorded")
        return "\n".join(lines)

    def write(self, *, json_path: str, log_path: str) -> None:
        """Write the machine JSON and the human log side-by-side."""
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(self.human_summary())
            handle.write("\n")


__all__ = [
    "SCHEMA_VERSION",
    "LINK_WITHIN_CARD",
    "LINK_CROSS_CARD",
    "LINK_UNKNOWN",
    "REQUIRED_TRANSFER_METRIC_FIELDS",
    "ple_metric_schema",
    "classify_link",
    "RankIdentity",
    "HostMemorySnapshot",
    "CollectiveTraceEntry",
    "LifecycleEvent",
    "ChunkProgress",
    "Heartbeat",
    "FirstFatal",
    "RunLog",
]
