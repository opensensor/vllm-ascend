# SPDX-License-Identifier: Apache-2.0
"""Cold native graph leases and optional diagnostic submission accounting.

By default replay calls the native graph directly, without observer callbacks.
Opt-in diagnostics record Python submissions, not completed device collectives.
The caller must serialize capture, replay and retirement on the owner thread.
Retirement first stops admission, then invokes an explicit device-wide drain,
then resets every native graph before allowing collective resources to close.
There is no destructor, fallback, failed-operation retry or implicit cleanup.
Capture validates every configured owner and counter and acquires native leases
before capture. Only after draining and resetting ALL graphs can leases release.
Diagnostic accounting visits only nonzero captured geometry/fallback deltas.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class _GraphLease:
    owner: object
    ticket: object
    released: bool = False


@dataclass
class _GraphRecord:
    graph: object
    before: tuple
    fallback_before: dict
    calls: tuple = ()
    fallback_calls: tuple = ()
    captured: bool = False
    failed: bool = False
    replays: int = 0
    replay_pending: bool = False
    reset: bool = False
    # Retain record-time owners even when a failed capture did not finish.
    retained_owners: tuple = field(default_factory=tuple)
    leases: list = field(default_factory=list)


def _counter(value):
    if type(value) is not int or value < 0:
        raise ValueError("graph accounting requires exact nonnegative counters")
    return value


class CollectiveGraphObserver:
    """One facade's strong registry for successful AND failed native captures."""

    def __init__(self, collective, *, track_replays=False):
        if type(track_replays) is not bool:
            raise ValueError("graph replay tracking must be an explicit bool")
        self.collective = collective
        self.track_replays = track_replays
        self._records = {}
        self._active_capture = None
        self._retiring = False
        self._retired = False
        self._failed = False

    @property
    def active_capture(self):
        return self._active_capture

    @property
    def retired(self):
        return self._retired

    def _require_live(self):
        if self._retiring or self._retired or self._failed or self.collective.state != "active":
            raise RuntimeError("collective graph ownership is retiring, retired or quarantined")

    def _record(self, graph):
        record = self._records.get(id(graph))
        if record is None or record.graph is not graph:
            raise RuntimeError("collective graph was not registered before capture")
        return record

    def _quarantine(self, record=None):
        self._failed = True
        if record is not None:
            record.failed = True
        self.collective.state = "quarantined"
        for owner in self.collective.owners.values():
            if owner.state != "closed":
                owner.state = "quarantined"
        # The current mapping can have drifted. Capture-time owners remain
        # strongly retained and must be quarantined even if their key vanished.
        for captured in self._records.values():
            for owner in captured.retained_owners:
                if owner.state != "closed":
                    owner.state = "quarantined"

    def capture_begin(self, graph):
        """Register strong ownership BEFORE entering the native capture context."""
        self._require_live()
        self.collective._context()
        if self._active_capture is not None or id(graph) in self._records:
            raise RuntimeError("nested or repeated collective graph capture is forbidden")
        if not callable(getattr(graph, "reset", None)) or not callable(getattr(graph, "replay", None)):
            raise ValueError("collective graph requires explicit native reset and replay methods")
        before = tuple((key, owner, _counter(owner.calls)) for key, owner in self.collective.owners.items())
        fallback_before = {key: _counter(value) for key, value in self.collective.fallbacks.items()}
        record = _GraphRecord(graph, before, fallback_before, retained_owners=tuple(owner for _, owner, _ in before))
        self._records[id(graph)] = record
        self._active_capture = graph
        try:
            for owner in record.retained_owners:
                ticket = owner.acquire_graph_lease()
                if not ticket:
                    raise RuntimeError("native graph lease acquisition returned a null ticket")
                record.leases.append(_GraphLease(owner, ticket))
        except BaseException:
            self._quarantine(record)
            self._active_capture = None
            raise

    def require_capture(self):
        """Reject raw, unregistered capture admission from the Torch facade."""
        self._require_live()
        if self._active_capture is None:
            raise RuntimeError("collective capture requires an explicitly registered graph owner")
        return self._active_capture

    @contextmanager
    def capture_scope(self, graph):
        """Optional wrapper convenience covering setup and native capture exit."""
        self.capture_begin(graph)
        try:
            yield graph
        except BaseException as error:
            self.capture_failed(graph, error)
            raise
        else:
            self.capture_end(graph)

    def capture_end(self, graph):
        self._require_live()
        record = self._record(graph)
        if self._active_capture is not graph:
            raise RuntimeError("collective graph capture end does not match active capture")
        try:
            if set(self.collective.owners) != {key for key, _, _ in record.before}:
                raise RuntimeError("fixed collective owners changed during graph capture")
            calls = []
            for key, owner, before in record.before:
                after = _counter(owner.calls)
                if self.collective.owners[key] is not owner or after < before:
                    raise RuntimeError("fixed collective owner or counter changed during graph capture")
                if after != before:
                    calls.append((key, owner, after - before))
            fallback_calls = []
            if not set(record.fallback_before) <= set(self.collective.fallbacks):
                raise RuntimeError("collective fallback counters disappeared during capture")
            for key, after in self.collective.fallbacks.items():
                after = _counter(after)
                before = record.fallback_before.get(key, 0)
                if after < before:
                    raise RuntimeError("collective fallback counter decreased during capture")
                if after != before:
                    fallback_calls.append((key, after - before))
            record.calls, record.fallback_calls = tuple(calls), tuple(fallback_calls)
            record.captured = True
            self._active_capture = None
        except BaseException:
            self._quarantine(record)
            self._active_capture = None
            raise

    def capture_failed(self, graph, error):
        """Retain graph even when the wrapper never assigned a cache entry."""
        record = self._record(graph)
        self._quarantine(record)
        if self._active_capture is graph:
            self._active_capture = None

    def _require_replayable(self, graph):
        self._require_live()
        record = self._record(graph)
        if self._active_capture is not None or not record.captured or record.failed or record.reset:
            raise RuntimeError("collective graph is not available for replay")
        return record

    def before_replay(self, graph):
        if not self.track_replays:
            raise RuntimeError("per-replay diagnostics are disabled; call the native graph directly")
        record = self._require_replayable(graph)
        if record.replay_pending:
            raise RuntimeError("collective graph replay is already pending")
        try:
            self.collective.validate_graph_replay()
        except BaseException:
            self._quarantine(record)
            raise
        record.replay_pending = True

    def replayed(self, graph):
        """Account once immediately AFTER actual graph.replay() returns."""
        if not self.track_replays:
            raise RuntimeError("per-replay diagnostics are disabled; call the native graph directly")
        record = self._require_replayable(graph)
        if not record.replay_pending:
            raise RuntimeError("collective graph replay was not admitted")
        try:
            owners, fallbacks = self.collective.owners, self.collective.fallbacks
            for key, owner, _ in record.calls:
                _counter(owner.calls)
                if owners.get(key) is not owner:
                    raise RuntimeError("captured collective owner was replaced before replay")
            for key, _ in record.fallback_calls:
                _counter(fallbacks[key])
            for _, owner, count in record.calls:
                owner.calls += count
            for key, count in record.fallback_calls:
                fallbacks[key] += count
            record.replays += 1
            record.replay_pending = False
        except BaseException:
            self._quarantine(record)
            raise

    def replay_failed(self, graph, error):
        record = self._record(graph)
        record.replay_pending = False
        self._quarantine(record)

    def retire(self, synchronize):
        """Stop replay, drain once, explicitly reset graphs, retain on failure."""
        if self._retired:
            return self.snapshot()
        self._require_live()
        if self._active_capture is not None:
            raise RuntimeError("cannot retire while native graph capture is active")
        if not callable(synchronize):
            raise ValueError("collective graph retirement requires an explicit drain callable")
        if any(record.replay_pending for record in self._records.values()):
            raise RuntimeError("cannot retire with a pending graph replay submission")
        self.collective._context()
        self._retiring = True
        try:
            synchronize()
            for record in self._records.values():
                if getattr(record.graph, "capture_token", None) is not None:
                    raise RuntimeError("native graph still owns an active capture token")
                record.graph.reset()
                record.reset = True
            # UpdatableGraph inherits native reset but retains Python task,
            # provider and capture-resource metadata. Drop only after ALL native
            # resets succeeded, while every graph/owner remains strongly held.
            for record in self._records.values():
                for name in ("tasks", "provider_sizes", "capture_resources"):
                    metadata = getattr(record.graph, name, None)
                    if metadata is not None:
                        metadata.clear()
            for record in self._records.values():
                for lease in record.leases:
                    lease.owner.release_graph_lease(lease.ticket)
                    lease.released = True
            self._retired = True
            return self.snapshot()
        except BaseException:
            self._quarantine()
            raise

    def snapshot(self):
        """Return pointer-free lifecycle evidence, never completion evidence."""
        records = tuple(self._records.values())
        return dict(
            registered_graphs=len(records),
            captured_graphs=sum(record.captured for record in records),
            failed_graphs=sum(record.failed for record in records),
            replay_submissions=sum(record.replays for record in records),
            captured_calls=sum(count for record in records for _, _, count in record.calls),
            replayed_calls=sum(record.replays * count for record in records for _, _, count in record.calls),
            reset_graphs=sum(record.reset for record in records),
            active_capture=self._active_capture is not None,
            retired=self._retired,
            lifecycle_only=True,
            tracking_enabled=self.track_replays,
            count_semantics="capture-and-replay-submissions" if self.track_replays else "capture-only",
            native_leases_acquired=sum(len(record.leases) for record in records),
            native_leases_released=sum(lease.released for record in records for lease in record.leases),
        )
