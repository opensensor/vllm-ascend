# SPDX-License-Identifier: Apache-2.0
"""Breakable segment ownership with CPU models of qualified upstream APIs.

Execute the actual helper/platform classes; fake only Torch/upstream capture
dependencies. No asynchronous execution, device math or NPU speed is qualified.
"""

import ast
import builtins
import dataclasses
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard.npu_collective_graph import CollectiveGraphObserver

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runtime():
    # Model only the qualified upstream capture boundaries, then execute the
    # actual helper and platform classes without their accelerator imports.
    state = SimpleNamespace(
        events=[],
        active=None,
        graphs=[],
        fail_begin=0,
        fail_end=0,
        fail_reset=0,
        custom=True,
        tracking=False,
        gc_enabled=True,
        breaks=0,
        runnable_error=None,
        imports=[],
        draft=False,
        forward_available=True,
    )

    class NativeGraph:
        def __init__(self):
            self.index = len(state.graphs)
            state.graphs.append(self)
            self.tasks, self.provider_sizes, self.capture_resources = ["owned"], {"owned": 1}, {"owned": 1}
            self.capture_token = None

        def capture_begin(self, **kwargs):
            active = state.observer.require_capture()
            assert self in active.native_graphs and active._current_graph is self
            assert all(owner.leases for owner in state.facade.owners.values())
            state.events.append(("begin", self.index, kwargs))
            if self.index + 1 == state.fail_begin:
                raise RuntimeError("partial native capture begin")

        def capture_end(self):
            state.events.append(("end", self.index))
            if self.index + 1 == state.fail_end:
                raise RuntimeError("partial native capture end")

        def replay(self):
            state.events.append(("native-replay", self.index))

        def reset(self):
            state.events.append(("reset", self.index))
            if self.index + 1 == state.fail_reset:
                raise RuntimeError("partial native reset")

    class Capture:
        """Relevant qualified upstream begin/end/list semantics, including bug."""

        def __init__(self, pool=None):
            self.pool, self.segments = pool, []
            self._num_graphs = self._num_eager_breaks = 0
            self._current_graph, self._capturing = None, False

        def __enter__(self):
            assert state.active is None
            state.active = self
            self._begin_segment()
            return self

        def __exit__(self, *exc):
            try:
                self._end_segment()
            finally:
                state.active = None

        def _begin_segment(self):
            graph = NativeGraph()
            graph.capture_begin()
            self._current_graph, self._capturing = graph, True

        def _end_segment(self):
            if not self._capturing:
                return
            self._current_graph.capture_end()
            self.segments.append(self._current_graph.replay)
            self._num_graphs += 1
            self._current_graph, self._capturing = None, False

        def add_eager(self, fn):
            self._end_segment()
            result = fn()
            self.segments.append(fn)
            self._num_eager_breaks += 1
            self._begin_segment()
            return result

        def replay(self):
            for fn in self.segments:
                fn()

        @property
        def num_graphs(self):
            return self._num_graphs

        @property
        def num_eager_breaks(self):
            return self._num_eager_breaks

    def make_owner():
        owner = SimpleNamespace(calls=0, state="active", leases=[])

        def acquire():
            ticket = object()
            owner.leases.append(ticket)
            state.events.append(("lease-acquire",))
            return ticket

        def release(ticket):
            assert all(("reset", graph.index) in state.events for graph in state.graphs)
            owner.leases.remove(ticket)
            state.events.append(("lease-release",))

        owner.acquire_graph_lease, owner.release_graph_lease = acquire, release
        return owner

    facade = state.facade = SimpleNamespace(
        state="active",
        owners={("float16", 4096): make_owner(), ("float16", 8192): make_owner()},
        fallbacks={},
        _context=lambda: None,
        validate_graph_replay=lambda: state.events.append(("validate-replay",)),
    )
    state.observer = facade.graph_observer = CollectiveGraphObserver(facade)
    offloader = SimpleNamespace(
        sync_prev_onload=lambda: state.events.append(("offloader-sync",)),
        join_after_forward=lambda: state.events.append(("offloader-join",)),
    )
    upstream = SimpleNamespace(
        BreakableCUDAGraphCapture=Capture,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(CUDAGraph=NativeGraph),
            accelerator=SimpleNamespace(empty_cache=lambda: state.events.append(("empty-cache",))),
        ),
        gc=SimpleNamespace(isenabled=lambda: state.gc_enabled, collect=lambda: state.events.append(("gc",))),
        validate_cudagraph_capturing_enabled=lambda: state.events.append(("validate-capture",)),
        set_graph_pool_id=lambda pool: state.events.append(("pool", pool)),
        current_platform=SimpleNamespace(graph_pool_handle=lambda: "global-pool"),
        get_offloader=lambda: offloader,
        weak_ref_tensors=lambda value: value,
        logger=SimpleNamespace(debug=lambda *args: None),
    )
    helper_namespace = dict(__name__="isolated_collective_breakable", upstream=upstream)
    helper_source = ROOT / "tools/sidecard/npu_collective_breakable.py"
    helper_ast = ast.parse(helper_source.read_text(), filename=str(helper_source))
    helper_ast.body = [node for node in helper_ast.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(helper_ast, str(helper_source), "exec"), helper_namespace)
    state.capture_type = helper_namespace["_CollectiveBreakableCapture"]

    class Mode(Enum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2

    @dataclasses.dataclass(frozen=True)
    class Descriptor:
        num_tokens: int = 1

    state.forward = SimpleNamespace(
        cudagraph_runtime_mode=Mode.FULL,
        capturing=False,
        batch_descriptor=Descriptor(),
        attn_metadata=None,
    )

    class WrapperBase:
        def __init__(self, runnable, vllm_config):
            self.runnable, self.vllm_config, self.graph_pool = runnable, vllm_config, "private-pool"
            self.entries = {}

        def __call__(self, *args, **kwargs):
            if state.forward.cudagraph_runtime_mode == Mode.NONE:
                return self.runnable(*args, **kwargs)
            entry = self.entries.setdefault(
                state.forward.batch_descriptor,
                SimpleNamespace(capture=None, output=None, batch_descriptor=state.forward.batch_descriptor),
            )
            if entry.capture is None:
                return self._capture(entry, args, kwargs)
            return self._replay(entry, args, kwargs)

        def _collect_tensor_addresses(self, args, kwargs):
            return [101, 202]

        def _capture(self, entry, args, kwargs):
            state.events.append(("original-upstream-capture",))
            entry.output = "baseline-output"
            entry.capture = SimpleNamespace(replay=lambda: state.events.append(("original-upstream-replay",)))
            return entry.output

        def _replay(self, entry, args, kwargs):
            entry.capture.replay()
            return entry.output

    def resolve(wrapper):
        if not state.custom:
            return None
        wrapper._collective_graph_observer = state.observer
        wrapper._collective_graph_tracking = state.observer.track_replays
        return state.observer

    def import_optional(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools.sidecard.npu_collective_breakable":
            state.imports.append(name)
            return SimpleNamespace(capture_with_collective=helper_namespace["capture_with_collective"])
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = dict(
        __name__=__name__,
        __builtins__={**vars(builtins), "__import__": import_optional},
        dataclass=dataclasses.dataclass,
        fields=dataclasses.fields,
        BreakableCUDAGraphWrapper=WrapperBase,
        BatchDescriptor=Descriptor,
        ACLGraphWrapper=SimpleNamespace(_resolve_collective_graph_observer=resolve),
        get_forward_context=lambda: state.forward,
        is_forward_context_available=lambda: state.forward_available,
        CUDAGraphMode=Mode,
        _EXTRA_CTX=SimpleNamespace(is_draft_model=False),
        torch=SimpleNamespace(
            npu=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(
                    synchronize=lambda: state.events.append(("stream-sync",)), wait_event=lambda event: None
                )
            )
        ),
        is_310p=lambda: False,
        qwen4exp_draft_supports_stream_ordering=lambda *args: False,
        qwen4exp_target_has_no_task_updates=lambda *args: False,
        get_graph_params=lambda: "main",
        get_draft_graph_params=lambda: "draft",
        get_draft_graph_prefill_params=lambda: "draft-prefill",
        weak_ref_workspaces=lambda value: state.events.append(("weak-workspace", value)),
    )
    source = ROOT / "vllm_ascend/compilation/breakable_aclgraph.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    parsed.body = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) or isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    exec(compile(parsed, str(source), "exec"), namespace)
    state.platform_namespace = namespace

    def runnable(*args, **kwargs):
        assert state.observer.require_capture() is state.active
        for index in range(state.breaks + 1):
            tuple(facade.owners.values())[0].calls += 1
            if state.runnable_error is not None:
                raise state.runnable_error
            if index < state.breaks:
                state.active.add_eager(lambda: state.events.append(("eager-break",)))
        return "custom-output"

    state.wrapper = namespace["BreakableACLGraphWrapper"](runnable, SimpleNamespace())
    return state


@pytest.mark.parametrize("breaks", (0, 1, 2, 4))
def test_every_segment_is_owned_before_native_begin_and_replay_stays_unobserved(runtime, breaks):
    state = runtime
    state.breaks = breaks
    assert state.wrapper() == "custom-output"
    entry = next(iter(state.wrapper.entries.values()))
    capture = entry.capture
    assert len(capture.native_graphs) == breaks + 1
    assert capture.num_eager_breaks == breaks and capture.num_graphs == breaks + 1
    assert entry.input_addresses == [101, 202]
    assert state.events.index(("lease-acquire",)) < next(
        i for i, event in enumerate(state.events) if event[0] == "begin"
    )
    assert [("weak-workspace", name) for name in ("main", "draft", "draft-prefill")] == state.events[-3:]
    if not breaks:
        assert capture.replay.__self__ is capture.native_graphs[0]
        assert capture.replay.__func__.__name__ == "replay"
    else:
        assert capture.replay.__func__.__name__ == "replay"
        assert capture.replay.__self__ is capture

    def forbidden(*args):
        pytest.fail("unobserved breakable replay entered Python diagnostics")

    state.observer.before_replay = state.observer.replayed = state.observer.replay_failed = forbidden
    state.facade.validate_graph_replay = forbidden
    before = tuple(state.facade.owners.values())[0].calls
    for _ in range(5):
        assert state.wrapper() == "custom-output"
    assert tuple(state.facade.owners.values())[0].calls == before
    assert sum(event[0] == "native-replay" for event in state.events) == 5 * (breaks + 1)
    assert sum(event[0] == "eager-break" for event in state.events) == 6 * breaks
    assert state.observer.snapshot()["replay_submissions"] == 0
    state.observer.retire(lambda: state.events.append(("device-drain",)))
    assert all(("reset", index) in state.events for index in range(breaks + 1))
    assert (
        state.observer.snapshot()["native_leases_acquired"] == state.observer.snapshot()["native_leases_released"] == 2
    )
    assert all(graph.tasks == [] for graph in capture.native_graphs)
    assert not capture.segments
    with pytest.raises(RuntimeError, match="retired"):
        state.wrapper()


@pytest.mark.parametrize("stage", ("first-begin", "later-begin", "end", "runnable"))
def test_partial_capture_failure_retains_all_native_segments_and_leases(runtime, stage):
    state = runtime
    state.breaks = 2
    if stage == "first-begin":
        state.fail_begin = 1
    elif stage == "later-begin":
        state.fail_begin = 2
    elif stage == "end":
        state.fail_end = 2
    else:
        state.runnable_error = KeyboardInterrupt("runnable capture failed")
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        state.wrapper()
    assert next(iter(state.wrapper.entries.values())).capture is None
    record = next(iter(state.observer._records.values()))
    assert record.graph.native_graphs == state.graphs
    assert state.graphs and not any(event[0] in ("reset", "lease-release") for event in state.events)
    assert all(owner.leases for owner in state.facade.owners.values())
    assert state.facade.state == "quarantined" and state.observer.snapshot()["failed_graphs"] == 1


def test_partial_reset_retains_all_segment_callables_and_every_lease(runtime):
    state = runtime
    state.breaks = 2
    state.wrapper()
    capture = next(iter(state.wrapper.entries.values())).capture
    state.fail_reset = 2
    with pytest.raises(RuntimeError, match="partial native reset"):
        state.observer.retire(lambda: None)
    assert capture.segments and all(graph.tasks for graph in capture.native_graphs)
    assert ("reset", 0) in state.events and ("reset", 1) in state.events and ("reset", 2) not in state.events
    assert not any(event[0] == "lease-release" for event in state.events)
    assert state.observer.snapshot()["native_leases_released"] == 0
    assert state.facade.state == "quarantined"


def test_baseline_keeps_original_upstream_capture_replay_without_optional_import(runtime):
    state = runtime
    state.custom = False
    assert state.wrapper() == state.wrapper() == "baseline-output"
    assert ("original-upstream-capture",) in state.events and ("original-upstream-replay",) in state.events
    assert not state.imports and not state.graphs
    assert state.observer.snapshot()["native_leases_acquired"] == 0


@pytest.mark.parametrize("gc_enabled,pool", ((True, "private-pool"), (False, None)))
def test_pre_capture_pool_gc_and_offloader_sequence_matches_qualified_upstream(runtime, gc_enabled, pool):
    state = runtime
    state.gc_enabled, state.wrapper.graph_pool = gc_enabled, pool
    state.wrapper()
    assert ("pool", pool or "global-pool") in state.events
    assert (("gc",) in state.events) is gc_enabled
    assert (("empty-cache",) in state.events) is gc_enabled
    assert state.events.index(("offloader-sync",)) < state.events.index(("lease-acquire",))
    assert state.events.index(("offloader-join",)) < state.events.index(("end", 0))
    begin = next(event for event in state.events if event[0] == "begin")
    assert begin[2] == (dict(pool=pool) if pool is not None else {})


def test_diagnostic_tracking_remains_explicit_and_rejects_eager_break_double_counting(runtime):
    state = runtime
    state.observer = state.facade.graph_observer = CollectiveGraphObserver(state.facade, track_replays=True)
    state.wrapper()
    for _ in range(3):
        state.wrapper()
    assert state.observer.snapshot()["replay_submissions"] == 3
    assert state.observer.snapshot()["replayed_calls"] == 3


def test_diagnostic_eager_break_mode_fails_closed_with_all_native_owners_retained(runtime):
    state = runtime
    state.breaks = 1
    state.observer = state.facade.graph_observer = CollectiveGraphObserver(state.facade, track_replays=True)
    with pytest.raises(RuntimeError, match="unobserved collective replay"):
        state.wrapper()
    assert state.facade.state == "quarantined" and len(state.graphs) == 2
    assert state.observer.snapshot()["native_leases_acquired"] == 2
    assert state.observer.snapshot()["native_leases_released"] == 0


def test_existing_draft_step_descriptor_identity_and_restore_are_preserved(runtime):
    state = runtime
    state.platform_namespace["_EXTRA_CTX"].is_draft_model = True
    first, second = object(), object()
    original = state.forward.batch_descriptor
    state.forward.draft_attn_metadatas = [first, second]
    for index, metadata in enumerate((first, second)):
        state.forward.attn_metadata = metadata
        assert state.wrapper() == "custom-output"
        assert state.forward.batch_descriptor is original
        assert any(getattr(key, "draft_step", None) == index for key in state.wrapper.entries)
    assert len(state.wrapper.entries) == 2
    assert state.observer.snapshot()["registered_graphs"] == 2
