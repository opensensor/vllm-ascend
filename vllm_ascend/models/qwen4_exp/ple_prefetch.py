# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async PLE row prefetch, de-duplication and metrics (plan T4.4, PRD R5).

The Qwen4Exp PLE table is one logical FP16 host copy (95.43 GiB) shared by the
four Ascend 310P workers. During decode every token requests several n-gram rows
(``[T, num_ngram_heads]``), and consecutive steps re-request the *same* hot rows
at a very high repeat rate. Fetching each row synchronously would stall the
decode loop with a host<->device sync per row, which the runtime forbids.

This module wraps the T4.1 host PLE table method
(:class:`~vllm_ascend.models.qwen4_exp.ngram_embedding.AscendPLEEmbeddingMethod`)
with an asynchronous, de-duplicating row prefetcher:

* **Batch + dedup** -- all requested ``(token, head)`` rows of a step are flattened
  and reduced to their unique ids in one shot, so a hot row is fetched at most
  once per step (dedup) and never at all when it is already resident (cache hit).
* **Async transfer, one batched wait** -- the unique *missing* rows are gathered on
  a dedicated side stream that overlaps with the preceding compute, exactly like
  the CUDA fork's ``Qwen4ExpPLEPinnedHostEmbedding.start_prefetch`` /
  ``_finalize_prefetch`` (side stream + a single ``wait_stream`` join). The decode
  loop issues **one** batched wait per step, never a wait per row.
* **Metrics for TOBS** -- every step emits host bytes, page faults, transfer
  bytes/step, hit rate and lookup latency in a stable schema
  (:data:`PLE_PREFETCH_METRIC_FIELDS`) that plan TOBS consumes.

There is no NPU on this dev path, so the async side stream is modelled by an
injectable :class:`PrefetchStream` abstraction with a simulated timeline
(:class:`SimulatedStream`). Submitted work runs only when the stream is joined,
which makes the overlap and the no-per-row-sync property directly assertable on
host. On device the same :class:`AscendRowPrefetcher` is driven by a stream
adapter backed by a real ``torch.npu.Stream`` (D1).

Scope: this module *wraps* the T4.1 gather API; it does not reimplement the host
table, and it does not edit the T4.1/T4.3 modules.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from .ngram_embedding import AscendPLEEmbeddingMethod

logger = logging.getLogger(__name__)

# Canonical metric schema consumed by plan TOBS. The five required run-log fields
# (host bytes, page faults, transfer bytes/step, hit rate, lookup latency) are
# listed first; the remaining fields give TOBS the counters those are derived
# from so the run log stays self-describing.
PLE_PREFETCH_METRIC_FIELDS: tuple[str, ...] = (
    "host_bytes",
    "page_faults",
    "transfer_bytes",
    "hit_rate",
    "lookup_latency",
    "requested_rows",
    "unique_rows",
    "cache_hits",
    "dedup_ratio",
)

# Default simulated transfer cost model (host sim only; real latency is measured
# on device at D1). A batched H2D transfer pays a fixed launch cost plus a
# per-byte streaming cost -- expressed in abstract "sim time" units.
_SIM_LAUNCH_COST = 1.0
_SIM_COST_PER_KIB = 0.05
_BYTES_PER_KIB = 1024


# --------------------------------------------------------------------------- #
# Injectable async-stream abstraction (host simulation of the NPU side stream)
# --------------------------------------------------------------------------- #


class PrefetchStream(Protocol):
    """Minimal async-stream contract the prefetcher drives.

    A real implementation wraps a ``torch.npu.Stream``: :meth:`submit` enqueues
    work that runs concurrently with compute, and :meth:`synchronize` is the
    single join that blocks until the enqueued work is done. The prefetcher only
    ever calls :meth:`synchronize` **once per decode step**, never per row.
    """

    def submit(self, fn: Callable[[], object], *, tag: str, cost: float = 0.0) -> int:
        """Enqueue ``fn`` on the side stream; return its handle. Non-blocking."""
        ...

    def synchronize(self, *, tag: str = "sync") -> list[object]:
        """Block once until all enqueued work finished; return results in order."""
        ...


@dataclass
class _StreamOp:
    tag: str
    cost: float
    payload: Callable[[], object]


class SimulatedStream:
    """Host simulation of a dedicated async transfer stream with a timeline.

    Submitted work is *not* executed until :meth:`synchronize` (the single
    batched wait) drains the queue -- so between a submit and the join the caller
    can do overlapping compute while the "transfer" is in flight. The recorded
    :attr:`timeline` and :attr:`wait_count` make the overlap and the
    no-per-row-sync property directly testable:

    * exactly one ``("wait", ...)`` entry per decode step, and
    * ``wait_count`` incremented by one per step regardless of row count.

    :attr:`clock` accumulates simulated latency (from each op's ``cost``) so a
    lookup latency can be reported without real hardware.
    """

    def __init__(self, name: str = "ple_prefetch") -> None:
        self.name = name
        self.timeline: list[tuple[str, str]] = []
        self.clock = 0.0
        self.submit_count = 0
        self.wait_count = 0
        self._pending: list[_StreamOp] = []

    @property
    def pending_count(self) -> int:
        """Number of ops enqueued but not yet drained by a join."""
        return len(self._pending)

    def submit(self, fn: Callable[[], object], *, tag: str, cost: float = 0.0) -> int:
        self._pending.append(_StreamOp(tag=tag, cost=float(cost), payload=fn))
        self.submit_count += 1
        self.timeline.append(("submit", tag))
        return len(self._pending) - 1

    def synchronize(self, *, tag: str = "sync") -> list[object]:
        # One blocking join drains everything enqueued since the last join. This
        # is the ONLY place the stream blocks, so a decode step that submits any
        # number of row transfers still incurs a single wait.
        self.wait_count += 1
        self.timeline.append(("wait", tag))
        results: list[object] = []
        for op in self._pending:
            self.clock += op.cost
            results.append(op.payload())
        self._pending.clear()
        return results

    def waits_in_timeline(self) -> int:
        """Count blocking joins recorded on the timeline (should equal steps)."""
        return sum(1 for kind, _ in self.timeline if kind == "wait")


# --------------------------------------------------------------------------- #
# Metrics (TOBS schema)
# --------------------------------------------------------------------------- #


@dataclass
class PLEPrefetchStepMetrics:
    """Per-step prefetch metrics (one decode step).

    ``dedup_ratio`` is the fraction of requested rows removed by de-duplication,
    ``(requested - unique) / requested``. ``hit_rate`` is the fraction of the
    *deduped* rows already resident (served without an H2D transfer),
    ``cache_hits / unique_rows``. ``page_faults`` is the number of unique rows
    that missed the resident cache and had to be transferred this step.
    """

    requested_rows: int
    unique_rows: int
    cache_hits: int
    page_faults: int
    transfer_bytes: int
    host_bytes: int
    lookup_latency: float

    @property
    def dedup_ratio(self) -> float:
        if self.requested_rows == 0:
            return 0.0
        return (self.requested_rows - self.unique_rows) / self.requested_rows

    @property
    def hit_rate(self) -> float:
        if self.unique_rows == 0:
            return 0.0
        return self.cache_hits / self.unique_rows

    def to_dict(self) -> dict:
        return {
            "host_bytes": self.host_bytes,
            "page_faults": self.page_faults,
            "transfer_bytes": self.transfer_bytes,
            "hit_rate": self.hit_rate,
            "lookup_latency": self.lookup_latency,
            "requested_rows": self.requested_rows,
            "unique_rows": self.unique_rows,
            "cache_hits": self.cache_hits,
            "dedup_ratio": self.dedup_ratio,
        }


@dataclass
class PLEPrefetchMetrics:
    """Aggregate prefetch metrics across decode steps (TOBS run-log schema).

    Mirrors the accounting style of
    :mod:`vllm_ascend.observability.qwen38_mem_accounting`: plain counters, a
    ``to_dict`` machine breakdown and a human summary. ``transfer_bytes`` here is
    the cumulative H2D bytes; per-step transfer bytes are kept in
    :attr:`transfer_bytes_per_step` so TOBS can chart transfer-bytes/step.
    """

    steps: int = 0
    requested_rows: int = 0
    unique_rows: int = 0
    cache_hits: int = 0
    page_faults: int = 0
    transfer_bytes: int = 0
    host_bytes: int = 0
    lookup_latency: float = 0.0
    transfer_bytes_per_step: list[int] = field(default_factory=list)

    def record_step(self, step: PLEPrefetchStepMetrics) -> None:
        self.steps += 1
        self.requested_rows += step.requested_rows
        self.unique_rows += step.unique_rows
        self.cache_hits += step.cache_hits
        self.page_faults += step.page_faults
        self.transfer_bytes += step.transfer_bytes
        self.lookup_latency += step.lookup_latency
        # host_bytes is a live footprint (resident cache), not a sum: the latest
        # step's value is authoritative.
        self.host_bytes = step.host_bytes
        self.transfer_bytes_per_step.append(step.transfer_bytes)

    @property
    def dedup_ratio(self) -> float:
        if self.requested_rows == 0:
            return 0.0
        return (self.requested_rows - self.unique_rows) / self.requested_rows

    @property
    def hit_rate(self) -> float:
        if self.unique_rows == 0:
            return 0.0
        return self.cache_hits / self.unique_rows

    @property
    def mean_transfer_bytes_per_step(self) -> float:
        if not self.transfer_bytes_per_step:
            return 0.0
        return sum(self.transfer_bytes_per_step) / len(self.transfer_bytes_per_step)

    def to_dict(self) -> dict:
        return {
            "host_bytes": self.host_bytes,
            "page_faults": self.page_faults,
            "transfer_bytes": self.transfer_bytes,
            "hit_rate": self.hit_rate,
            "lookup_latency": self.lookup_latency,
            "requested_rows": self.requested_rows,
            "unique_rows": self.unique_rows,
            "cache_hits": self.cache_hits,
            "dedup_ratio": self.dedup_ratio,
            "steps": self.steps,
            "mean_transfer_bytes_per_step": self.mean_transfer_bytes_per_step,
            "transfer_bytes_per_step": list(self.transfer_bytes_per_step),
        }

    def human_summary(self) -> str:
        return (
            f"PLE prefetch: steps={self.steps} "
            f"requested={self.requested_rows} unique={self.unique_rows} "
            f"dedup={self.dedup_ratio:.2%} hit_rate={self.hit_rate:.2%} "
            f"page_faults={self.page_faults} "
            f"transfer_bytes={self.transfer_bytes} "
            f"host_bytes={self.host_bytes} "
            f"lookup_latency={self.lookup_latency:.3f}"
        )


# --------------------------------------------------------------------------- #
# Prefetch handle + prefetcher
# --------------------------------------------------------------------------- #


@dataclass
class PrefetchHandle:
    """Opaque handle tying a submitted transfer to its assembly on :meth:`wait`."""

    original_shape: tuple[int, ...]
    unique_ids: torch.Tensor
    inverse: torch.Tensor
    missing_ids: torch.Tensor
    cached_ids: torch.Tensor
    step: PLEPrefetchStepMetrics
    _submitted: bool


class AscendRowPrefetcher:
    """Async, de-duplicating row prefetcher over a T4.1 host PLE table method.

    The prefetcher keeps a small **resident cache** of recently fetched rows (an
    LRU of row vectors) so that repeated hot rows across steps become cache hits
    (no transfer). Per decode step:

    1. :meth:`prefetch_step` flattens the requested ids, dedups them, splits them
       into cache *hits* and *misses* (page faults), and submits **one** batched
       async gather of the missing rows on the side stream -- returning
       immediately so the caller overlaps compute.
    2. :meth:`wait` performs the **single** batched join, installs the fetched
       rows into the cache, reassembles the full ``[*ids, dim]`` result and emits
       :class:`PLEPrefetchStepMetrics`.

    :meth:`decode_step` is the convenience wrapper that runs a compute callback
    *between* the submit and the join, so the overlap is exercised directly.

    Args:
        ple_method: the T4.1 host PLE table method (provides ``gather_rows``,
            ``embedding_dim``, ``itemsize``, ``dtype``). Duck-typed so both host
            transports work unchanged.
        stream: injectable async stream (defaults to a fresh
            :class:`SimulatedStream`).
        cache_capacity_rows: max resident rows kept for cross-step reuse. ``None``
            keeps every fetched row (unbounded); a positive int caps the resident
            footprint with LRU eviction. ``0`` disables the cache (every unique
            row is always a page fault).
        latency_model: ``(transfer_bytes, num_missing_rows) -> latency`` used for
            the simulated lookup-latency metric. Defaults to a launch + per-byte
            model. On device this is replaced by the measured transfer time.
        metrics: an existing aggregate to accumulate into (created if omitted).
    """

    def __init__(
        self,
        ple_method: AscendPLEEmbeddingMethod,
        *,
        stream: PrefetchStream | None = None,
        cache_capacity_rows: int | None = None,
        latency_model: Callable[[int, int], float] | None = None,
        metrics: PLEPrefetchMetrics | None = None,
    ) -> None:
        if cache_capacity_rows is not None and cache_capacity_rows < 0:
            raise ValueError("cache_capacity_rows must be >= 0 or None")
        self.ple_method = ple_method
        self.stream: PrefetchStream = stream if stream is not None else SimulatedStream()
        self.cache_capacity_rows = cache_capacity_rows
        self.latency_model = latency_model or _default_latency_model
        self.metrics = metrics if metrics is not None else PLEPrefetchMetrics()
        self.embedding_dim = int(ple_method.embedding_dim)
        self.row_bytes = self.embedding_dim * int(ple_method.itemsize)
        # Resident cache: row id -> row vector, ordered by recency (LRU tail).
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    # -- introspection ----------------------------------------------------- #

    @property
    def resident_rows(self) -> int:
        return len(self._cache)

    @property
    def host_bytes(self) -> int:
        """Live host footprint of the resident staging cache (one shared copy)."""
        return self.resident_rows * self.row_bytes

    def resident_ids(self) -> set[int]:
        return set(self._cache.keys())

    # -- decode step ------------------------------------------------------- #

    def prefetch_step(self, ngram_ids: torch.Tensor) -> PrefetchHandle:
        """Dedup requested rows and submit one async transfer of the misses.

        Returns immediately (no join) so the caller can overlap compute before
        calling :meth:`wait`.
        """
        original_shape = tuple(ngram_ids.shape)
        flat = ngram_ids.reshape(-1).long()
        requested = int(flat.numel())

        # Batched dedup: unique ids + inverse index to rebuild the full request.
        unique_ids, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        unique_count = int(unique_ids.numel())

        resident = self._cache
        if resident:
            unique_list = unique_ids.tolist()
            missing_list = [rid for rid in unique_list if rid not in resident]
        else:
            missing_list = unique_ids.tolist()
        missing_ids = torch.as_tensor(missing_list, dtype=torch.long)
        cached_count = unique_count - len(missing_list)
        transfer_bytes = len(missing_list) * self.row_bytes
        latency = float(self.latency_model(transfer_bytes, len(missing_list)))

        submitted = len(missing_list) > 0
        if submitted:
            # ONE batched async gather of all missing rows -- never per row.
            self.stream.submit(
                lambda ids=missing_ids: self.ple_method.gather_rows(ids),
                tag=f"gather:{len(missing_list)}",
                cost=latency,
            )

        step = PLEPrefetchStepMetrics(
            requested_rows=requested,
            unique_rows=unique_count,
            cache_hits=cached_count,
            page_faults=len(missing_list),
            transfer_bytes=transfer_bytes,
            host_bytes=0,  # filled in on wait() after the cache is updated
            lookup_latency=latency,
        )
        return PrefetchHandle(
            original_shape=original_shape,
            unique_ids=unique_ids,
            inverse=inverse,
            missing_ids=missing_ids,
            cached_ids=unique_ids,
            step=step,
            _submitted=submitted,
        )

    def wait(self, handle: PrefetchHandle) -> torch.Tensor:
        """Single batched join, install fetched rows, reassemble the result.

        Returns the gathered embeddings shaped ``[*original_shape, dim]`` and
        records the step metrics on :attr:`metrics`.
        """
        # THE single blocking wait for the whole step (a batched join, not one
        # wait per row).
        results = self.stream.synchronize(tag="ple_prefetch")
        if handle._submitted:
            fetched = results[-1]
            if not isinstance(fetched, torch.Tensor):
                raise TypeError("prefetch stream returned a non-tensor gather result")
            missing_list = handle.missing_ids.tolist()
            for i, rid in enumerate(missing_list):
                self._cache_put(int(rid), fetched[i])

        # Reassemble unique rows from the (now complete) resident cache, then
        # expand back to the full request via the inverse index.
        unique_list = handle.unique_ids.tolist()
        if unique_list:
            unique_rows = torch.stack([self._cache_get(int(rid)) for rid in unique_list])
        else:
            unique_rows = self._empty_rows()
        flat_rows = unique_rows.index_select(0, handle.inverse)
        out = flat_rows.reshape(*handle.original_shape, self.embedding_dim)

        step = handle.step
        step.host_bytes = self.host_bytes
        self.metrics.record_step(step)
        return out

    def decode_step(
        self,
        ngram_ids: torch.Tensor,
        compute_fn: Callable[[], object] | None = None,
    ) -> torch.Tensor:
        """Prefetch, run ``compute_fn`` overlapped with the transfer, then join.

        This is the shape of the real decode loop: the row transfer is launched,
        the layer's compute runs while it is in flight, and a single batched wait
        collects the rows. ``compute_fn`` runs while the stream still has work
        pending (asserted testable via the simulated timeline).
        """
        handle = self.prefetch_step(ngram_ids)
        if compute_fn is not None:
            compute_fn()
        return self.wait(handle)

    # -- cache internals --------------------------------------------------- #

    def _cache_put(self, row_id: int, row: torch.Tensor) -> None:
        if self.cache_capacity_rows == 0:
            return
        cache = self._cache
        cache[row_id] = row.detach().clone()
        cache.move_to_end(row_id)
        if self.cache_capacity_rows is not None:
            while len(cache) > self.cache_capacity_rows:
                cache.popitem(last=False)

    def _cache_get(self, row_id: int) -> torch.Tensor:
        cache = self._cache
        row = cache.get(row_id)
        if row is None:
            # Disabled/evicted cache: fall back to a direct batched gather of the
            # single id (still no per-row *stream* sync -- this is a host lookup).
            return self.ple_method.gather_rows(torch.as_tensor([row_id], dtype=torch.long))[0]
        cache.move_to_end(row_id)
        return row

    def _empty_rows(self) -> torch.Tensor:
        weight = getattr(self.ple_method, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight.new_empty((0, self.embedding_dim))
        return torch.empty((0, self.embedding_dim), dtype=self.ple_method.dtype)


def _default_latency_model(transfer_bytes: int, num_rows: int) -> float:
    """Simulated H2D lookup latency: fixed launch + per-KiB streaming cost.

    Returns 0.0 for an all-hit step (no transfer). Host simulation only; the real
    latency is measured on the side stream at D1.
    """
    if num_rows == 0:
        return 0.0
    return _SIM_LAUNCH_COST + _SIM_COST_PER_KIB * (transfer_bytes / _BYTES_PER_KIB)


__all__ = [
    "PLE_PREFETCH_METRIC_FIELDS",
    "AscendRowPrefetcher",
    "PLEPrefetchMetrics",
    "PLEPrefetchStepMetrics",
    "PrefetchHandle",
    "PrefetchStream",
    "SimulatedStream",
]
