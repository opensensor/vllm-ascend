# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from tools.sidecard.execution_trace import ExecutionTrace, options
from tools.sidecard.trace_summary import summarize


class Backend:
    def __init__(self):
        self.events, self.ready, self.queries = [], False, 0

    def Event(self, *, enable_timing):
        assert enable_timing
        backend = self

        class Event:
            def record(self):
                pass

            def query(self):
                backend.queries += 1
                return backend.ready

            def elapsed_time(self, other):
                assert backend.ready
                return 2.5

            def synchronize(self):
                raise AssertionError("trace must not add synchronization")

        event = Event()
        self.events.append(event)
        return event


def enabled(**kwargs):
    return dict(execution_trace=dict(enabled=True, run_id="test", **kwargs))


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        {"enabled": 1},
        {"max_records": 0},
        {"max_records": 65537},
        {"sample_every": False},
        {"sample_every": 0},
        {"run_id": 0},
        {"run_id": "x" * 257},
        {"extra": True},
    ],
)
def test_invalid_options(value):
    with pytest.raises(ValueError):
        options(dict(execution_trace=value))


def test_default_off_never_touches_backend_or_writes(tmp_path):
    trace = ExecutionTrace({}, tmp_path, "cuda", SimpleNamespace())
    trace.begin(0, 0, 1)
    with trace.span("noop", device=True):
        pass
    trace.collect_ready()
    trace.close()
    assert not list(tmp_path.iterdir())
    assert not trace.records


def test_pending_events_nonblocking_and_explicit_clock_domains(tmp_path):
    backend = Backend()
    trace = ExecutionTrace(enabled(), tmp_path, "npu-rank0", backend)
    trace.begin(3, 0, 1)
    with trace.span("stage_execution", device=True, layers=[18, 19]):
        pass
    trace.flush()
    assert trace.records[0]["device_ms"] is None
    assert len(trace.pending) == 1
    header = json.loads(trace.output.read_text().splitlines()[0])
    assert header["device_to_host_mapping"] is None
    assert header["unresolved_device_events"] == 1
    backend.ready = True
    trace.close()
    assert trace.records[0]["device_ms"] == 2.5
    assert not trace.pending
    assert trace.records[0]["device_clock"].startswith("owner-local:npu-rank0:")


def test_nested_spans_bounded_even_before_outer_exit(tmp_path):
    backend = Backend()
    trace = ExecutionTrace(enabled(max_records=2), tmp_path, "cuda", backend)
    trace.begin(0, 0, 1)
    with trace.span("iteration", device=True), trace.span("prefix", device=True), trace.span("too_many", device=True):
        pass
    assert len(trace.records) == 2 and len(trace.pending) == 2
    assert len(backend.events) == 4
    assert trace.dropped == 1


def test_sampling_and_request_identity_survive_prefill_revisits(tmp_path):
    trace = ExecutionTrace(enabled(sample_every=2), tmp_path, "cuda")
    for sequence, position in [(0, 0), (1, 32), (0, 0), (2, 64), (3, 0), (4, 32)]:
        trace.begin(sequence, position, 32)
        with trace.span("iteration"):
            pass
    assert [r["sequence"] for r in trace.records] == [0, 0, 2, 4]
    assert [r["request"] for r in trace.records] == [0, 0, 0, 1]


def test_exception_keeps_failed_span_without_sync(tmp_path):
    trace = ExecutionTrace(enabled(), tmp_path, "cuda", Backend())
    trace.begin(0, 0, 1)
    with pytest.raises(RuntimeError), trace.span("failed", device=True):
        raise RuntimeError("test")
    assert trace.records[0]["failed"]
    assert trace.records[0]["host_end_ns"] >= trace.records[0]["host_start_ns"]


def test_summary_joins_only_whole_stage_identity_and_preserves_overlap(tmp_path):
    paths = []
    for owner in ("cuda", "npu-rank0", "npu-rank1"):
        clock = iter(range(1000, 10000, 100)).__next__
        trace = ExecutionTrace(enabled(), tmp_path, owner, clock=clock)
        trace.begin(10, 0, 1)
        with (
            trace.span("iteration" if owner == "cuda" else "stage_execution"),
            trace.span("prefix" if owner == "cuda" else "host_acknowledgement"),
        ):
            pass
        trace.close()
        paths.append(trace.output)
    result = summarize(paths)
    assert len(result["iterations"]) == 1
    row = result["iterations"][0]
    assert row["participating_owners"] == ["cuda", "npu-rank0", "npu-rank1"]
    assert row["observed_iteration_ms"] is not None
    assert len(row["timeline"]) == 6
    assert len(row["observed_host_dependency_chain"]) == 1
    # Same counter from a different run or host cannot be paired.
    contents = paths[2].read_text().splitlines()
    header = json.loads(contents[0])
    header["run_id"] = "different-run"
    paths[2].write_text(json.dumps(header) + "\n" + "\n".join(contents[1:]) + "\n")
    assert len(summarize(paths)["iterations"]) == 2


def test_summary_rejects_legacy_independent_expert_receipts_and_duplicates(tmp_path):
    bad = tmp_path / "expert.json"
    bad.write_text(json.dumps({"sequence": 0, "rank": 0}) + "\n")
    with pytest.raises(ValueError, match="whole-stage"):
        summarize([bad])
    trace = ExecutionTrace(enabled(), tmp_path, "cuda")
    trace.begin(0, 0, 1)
    with trace.span("iteration"):
        pass
    trace.close()
    with pytest.raises(ValueError, match="duplicate"):
        summarize([trace.output, trace.output])


@pytest.mark.parametrize(
    "value,reason",
    [
        (-27.947, "negative_device_interval"),
        (float("nan"), "nonfinite_device_interval"),
        (float("inf"), "nonfinite_device_interval"),
        (float("-inf"), "nonfinite_device_interval"),
        ("not a duration", "nonnumeric_device_interval"),
        (True, "nonnumeric_device_interval"),
    ],
)
def test_invalid_device_samples_unavailable_without_blocking_cleanup(tmp_path, value, reason):
    backend = Backend()
    backend.ready = True
    trace = ExecutionTrace(enabled(), tmp_path, "npu-rank0", backend)
    trace.begin(0, 0, 1)
    with trace.span("eager_span", device=True, layers=[23]):
        pass
    backend.events[0].elapsed_time = lambda other: value
    trace.close()
    assert trace.closed and not trace.pending
    record = trace.records[0]
    assert record["device_ms"] is None
    assert record["device_timing_error"] == dict(kind=reason, raw_value=repr(value))
    assert record["host_end_ns"] >= record["host_start_ns"]
    header = json.loads(trace.output.read_text().splitlines()[0])
    assert header["invalid_device_timings"] == 1
    assert header["device_timing_errors"] == {reason: 1}
    assert header["device_timing_reliability"] == "unreliable"
    # The on-disk evidence remains strict JSON, including nonfinite failures.
    json.loads(trace.output.read_text().splitlines()[1], parse_constant=lambda value: pytest.fail(value))


@pytest.mark.parametrize("phase", ["query", "elapsed"])
def test_event_api_failure_is_diagnostic_and_not_retried_on_cleanup(tmp_path, phase):
    backend = Backend()
    backend.ready = True
    trace = ExecutionTrace(enabled(), tmp_path, "npu-rank1", backend)
    trace.begin(0, 0, 1)
    with trace.span("stage_execution", device=True):
        pass
    failures = []

    def fail(*args):
        failures.append(phase)
        raise RuntimeError("event timestamp unavailable")

    event, attribute = (backend.events[1], "query") if phase == "query" else (backend.events[0], "elapsed_time")
    setattr(event, attribute, fail)
    trace.collect_ready()
    trace.close()
    assert failures == [phase]
    assert not trace.pending
    record = trace.records[0]
    assert record["device_ms"] is None
    assert record["device_timing_error"] == dict(
        kind=f"device_event_{phase}_failed", exception_type="RuntimeError", message="event timestamp unavailable"
    )
    summary = summarize([trace.output])
    assert summary["invalid_device_timings"] == 1
    assert summary["device_timing_reliability"] == "unreliable"


@pytest.mark.parametrize("value", [-38.154, float("nan"), float("inf"), float("-inf")])
def test_archived_invalid_samples_flagged_without_altering_host_spans(tmp_path, value):
    backend = Backend()
    backend.ready = True
    trace = ExecutionTrace(enabled(), tmp_path, "npu-rank1", backend)
    trace.begin(0, 0, 1)
    with trace.span("eager_span", device=True, layers=[31]):
        pass
    with trace.span("result_packing", device=True):
        pass
    trace.close()
    records = [json.loads(line) for line in trace.output.read_text().splitlines()]
    header = records[0]
    for name in ("invalid_device_timings", "device_timing_errors", "device_timing_reliability"):
        header.pop(name)
    records[1]["device_ms"] = value
    trace.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    archived = trace.output.read_bytes()
    summary = summarize([trace.output])
    assert summary["invalid_device_timings"] == 1
    assert summary["metadata"][0]["invalid_device_timings"] == 1
    assert summary["metadata"][0]["device_timing_reliability"] == "unreliable"
    timeline = summary["iterations"][0]["timeline"]
    assert timeline[0]["device_ms"] is None
    assert timeline[0]["device_timing_error"]["raw_value"] == repr(value)
    assert timeline[0]["host_ms"] == (records[1]["host_end_ns"] - records[1]["host_start_ns"]) / 1e6
    assert timeline[1]["device_ms"] == 2.5
    assert "nonfinite" in summary["limitations"][-1]
    # Do not rewrite raw evidence when summarizing an earlier trace format.
    assert trace.output.read_bytes() == archived
    json.dumps(summary, allow_nan=False)
