# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in host spans and asynchronous owner-local device events.

Runtime JSON accepts ``execution_trace`` with enabled (default false),
max_records (1..65536), sample_every (positive integer), and run_id (string).
Events are collected only when query reports completion; this module never
synchronizes a device. Device event durations have no host timestamp mapping.
Files are written at explicit flush/close boundaries, outside operator loops.
"""

import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_RECORD_LIMIT = 4096
MAX_RECORD_LIMIT = 65536
SCHEMA = "sidecard-execution-trace-v1"


def device_interval_error(value):
    """Reject impossible timing samples without asserting that others are sound."""
    if type(value) not in (int, float):
        reason = "nonnumeric_device_interval"
    elif not math.isfinite(value):
        reason = "nonfinite_device_interval"
    elif value < 0:
        reason = "negative_device_interval"
    else:
        return None
    return dict(kind=reason, raw_value=repr(value)[:128])


def unavailable_device_interval(record, error):
    record["device_ms"] = None
    record["device_timing_error"] = error


def timing_error_counts(records):
    counts = {}
    for record in records:
        error = record.get("device_timing_error")
        if error is not None:
            kind = error["kind"]
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def options(spec):
    value = spec.get("execution_trace", {})
    if not isinstance(value, dict) or set(value) - {"enabled", "max_records", "sample_every", "run_id"}:
        raise ValueError("invalid execution_trace runtime options")
    result = dict(enabled=False, max_records=DEFAULT_RECORD_LIMIT, sample_every=1, run_id="") | value
    if type(result["enabled"]) is not bool:
        raise ValueError("execution_trace.enabled must be boolean")
    if type(result["max_records"]) is not int or not 1 <= result["max_records"] <= MAX_RECORD_LIMIT:
        raise ValueError("execution_trace.max_records must be within 1..65536")
    if type(result["sample_every"]) is not int or result["sample_every"] < 1:
        raise ValueError("execution_trace.sample_every must be positive")
    if not isinstance(result["run_id"], str) or len(result["run_id"]) > 256:
        raise ValueError("execution_trace.run_id must be a bounded string")
    return result


class ExecutionTrace:
    def __init__(self, config, output, owner, backend=None, *, clock=time.monotonic_ns):
        self.options = options(config)
        self.enabled = self.options["enabled"]
        self.output = Path(output) / f"execution-trace-{owner}.jsonl"
        self.owner, self.backend, self.clock = owner, backend, clock
        self.run_id = self.options["run_id"] or Path(output).name
        self.records, self.pending = [], []
        self.identity = None
        self.request, self.last_sequence = -1, None
        self.last_request_sequence = -1
        self.dropped = self.instrumentation_ns = 0
        self.closed = False
        self.host_clock = "host-monotonic:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def begin(self, sequence, position, rows):
        if self.enabled:
            if position == 0 and sequence > self.last_request_sequence:
                self.request += 1
                self.last_request_sequence = sequence
            self.last_sequence = sequence
            self.identity = dict(request=self.request, sequence=sequence, position=position, rows=rows)

    @contextmanager
    def span(self, name, *, device=False, layer=None, layers=None, attribution=None):
        if not self.enabled or self.identity is None or self.identity["sequence"] % self.options["sample_every"]:
            yield
            return
        if len(self.records) >= self.options["max_records"]:
            self.dropped += 1
            yield
            return
        setup = self.clock()
        record = dict(self.identity, name=name, layer=layer, layers=layers, attribution=attribution)
        events = None
        if device:
            if self.backend is None:
                raise ValueError("device tracing requires an owner-local event backend")
            events = (self.backend.Event(enable_timing=True), self.backend.Event(enable_timing=True))
            events[0].record()
        record["host_start_ns"] = self.clock()
        self.instrumentation_ns += record["host_start_ns"] - setup
        # Reserve before entering nested spans so the event/storage limit is hard.
        self.records.append(record)
        try:
            yield
        except BaseException:
            record["failed"] = True
            raise
        finally:
            record["host_end_ns"] = self.clock()
            finish = record["host_end_ns"]
            if events is not None:
                events[1].record()
                record["device_clock"] = f"owner-local:{self.owner}:{os.getpid()}"
                record["device_ms"] = None
                self.pending.append((record, *events))
            self.instrumentation_ns += self.clock() - finish

    def collect_ready(self):
        """Nonblocking query only, normally called after an existing wait."""
        if not self.enabled:
            return
        start = self.clock()
        pending = []
        for record, begin, end in self.pending:
            try:
                ready = end.query()
            except Exception as error:
                unavailable_device_interval(
                    record,
                    dict(
                        kind="device_event_query_failed", exception_type=type(error).__name__, message=str(error)[:256]
                    ),
                )
                continue
            if not ready:
                pending.append((record, begin, end))
                continue
            try:
                value = begin.elapsed_time(end)
            except Exception as error:
                unavailable_device_interval(
                    record,
                    dict(
                        kind="device_event_elapsed_failed",
                        exception_type=type(error).__name__,
                        message=str(error)[:256],
                    ),
                )
                continue
            error = device_interval_error(value)
            if error is None:
                record["device_ms"] = value
            else:
                unavailable_device_interval(record, error)
        self.pending = pending
        self.instrumentation_ns += self.clock() - start

    def flush(self):
        if not self.enabled:
            return
        self.collect_ready()
        errors = timing_error_counts(self.records)
        header = dict(
            schema=SCHEMA,
            kind="metadata",
            run_id=self.run_id,
            owner=self.owner,
            pid=os.getpid(),
            host_clock=self.host_clock,
            options=self.options,
            instrumentation_ns=self.instrumentation_ns,
            overhead_measurement="host instrumentation API cost only; wall-time slowdown requires paired runs",
            dropped=self.dropped,
            unresolved_device_events=len(self.pending),
            invalid_device_timings=sum(errors.values()),
            device_timing_errors=errors,
            device_timing_reliability="unreliable" if errors else "unverified",
            device_to_host_mapping=None,
            collective_attribution="unavailable: graph-internal arrival wait and transit not separable",
            coverage="complete-layer actor/DMA path only; expert-only runtime unavailable",
        )
        with self.output.open("w") as handle:
            handle.write(json.dumps(header) + "\n")
            for record in self.records:
                handle.write(json.dumps(dict(record, kind="span")) + "\n")

    def close(self):
        if not self.closed:
            self.flush()
            self.closed = True
