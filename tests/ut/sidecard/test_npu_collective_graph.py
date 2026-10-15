# SPDX-License-Identifier: Apache-2.0
"""Portable lifecycle and actual-wrapper hook tests, without importing Torch."""

import ast
import builtins
import dataclasses
import gc
import json
import weakref
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.sidecard.npu_collective_graph import CollectiveGraphObserver

ROOT = Path(__file__).resolve().parents[3]


def install_leases(owner):
    owner.leases = []

    def acquire():
        ticket = object()
        owner.leases.append(ticket)
        return ticket

    def release(ticket):
        owner.leases.remove(ticket)

    owner.acquire_graph_lease, owner.release_graph_lease = acquire, release
    return owner


class Graph:
    def __init__(self):
        self.events = []
        self.tasks, self.provider_sizes, self.capture_resources = ["task"], {"provider": 1}, {"resource": "owned"}
        self.capture_token = None
        self.reset_error = self.replay_error = self.update_error = None

    def replay(self):
        self.events.append("replay")
        if self.replay_error:
            raise self.replay_error
        return "submitted"

    def reset(self):
        self.events.append("reset")
        if self.reset_error:
            raise self.reset_error

    def resolve_tasks(self, source):
        return ()

    def update(self, stream, tasks):
        self.events.append("update")
        if self.update_error:
            raise self.update_error


@pytest.fixture
def collective():
    owner = install_leases(SimpleNamespace(calls=5, state="active"))
    second = install_leases(SimpleNamespace(calls=2, state="active"))
    facade = SimpleNamespace(
        state="active",
        owners={("float16", 4096): owner, ("float32", 32): second},
        fallbacks={("torch.bfloat16", 17, True): 3},
        validations=[],
        validation_error=None,
    )

    def context():
        facade.validations.append("context")
        if facade.validation_error:
            raise facade.validation_error

    def validate_replay():
        facade.validations.append("replay-context-and-stream")
        if facade.validation_error:
            raise facade.validation_error

    facade._context, facade.validate_graph_replay = context, validate_replay
    facade.graph_observer = CollectiveGraphObserver(facade, track_replays=True)
    return facade


def capture(facade, graph, counts=(3, 1), fallback=2):
    observer = facade.graph_observer
    with observer.capture_scope(graph):
        assert observer.require_capture() is observer.active_capture is graph
        for owner, count in zip(facade.owners.values(), counts):
            owner.calls += count
        facade.fallbacks["torch.bfloat16", 17, True] += fallback


def replay(observer, graph):
    observer.before_replay(graph)
    graph.replay()
    observer.replayed(graph)


def test_capture_and_replay_account_once_per_graph_including_multiple_geometries(collective):
    facade, graph = collective, Graph()
    observer = facade.graph_observer
    capture(facade, graph)
    owners = tuple(facade.owners.values())
    for _ in range(5):
        replay(observer, graph)
    assert [owner.calls for owner in owners] == [23, 8]
    assert facade.fallbacks["torch.bfloat16", 17, True] == 15
    assert facade.validations == ["context"] + ["replay-context-and-stream"] * 5
    assert observer.snapshot() == dict(
        registered_graphs=1,
        captured_graphs=1,
        failed_graphs=0,
        replay_submissions=5,
        captured_calls=4,
        replayed_calls=20,
        reset_graphs=0,
        active_capture=False,
        retired=False,
        lifecycle_only=True,
        tracking_enabled=True,
        count_semantics="capture-and-replay-submissions",
        native_leases_acquired=2,
        native_leases_released=0,
    )
    assert json.loads(json.dumps(observer.snapshot())) == observer.snapshot()


def test_records_each_capture_delta_without_crediting_eager_calls(collective):
    facade, first, second = collective, Graph(), Graph()
    observer = facade.graph_observer
    capture(facade, first, (1, 0), 0)
    tuple(facade.owners.values())[0].calls += 7  # unrelated eager work
    capture(facade, second, (0, 3), 1)
    replay(observer, first)
    replay(observer, second)
    replay(observer, second)
    assert [owner.calls for owner in facade.owners.values()] == [14, 11]
    assert observer.snapshot()["captured_calls"] == 4
    assert observer.snapshot()["replayed_calls"] == 7


def test_capture_keeps_full_owners_but_hot_accounting_touches_only_nonzero_deltas(collective):
    class GuardedOwner:
        def __init__(self):
            self.state, self.guard, self._calls = "active", False, 0

        @property
        def calls(self):
            if self.guard:
                pytest.fail("replay read an unrelated geometry counter")
            return self._calls

        @calls.setter
        def calls(self, value):
            if self.guard:
                pytest.fail("replay wrote an unrelated geometry counter")
            self._calls = value

    class KeyedOwners(dict):
        def values(self):
            pytest.fail("replay scanned every geometry instead of its captured key")

    collective.owners = KeyedOwners(
        {("float16", 4096 * (index + 1)): install_leases(GuardedOwner()) for index in range(14)}
    )
    collective.fallbacks = {("torch.bfloat16", index, True): 1 for index in range(14)}
    captured_key = ("float16", 4096 * 14)
    fallback_key = ("torch.bfloat16", 13, True)
    graph, observer = Graph(), collective.graph_observer
    observer.capture_begin(graph)
    collective.owners[captured_key].calls += 3
    collective.fallbacks[fallback_key] += 2
    observer.capture_end(graph)
    record = observer._records[id(graph)]
    assert len(record.before) == len(record.retained_owners) == 14
    assert record.calls == ((captured_key, collective.owners[captured_key], 3),)
    assert record.fallback_calls == ((fallback_key, 2),)
    for key, owner in collective.owners.items():
        owner.guard = key != captured_key
    for _ in range(5):
        replay(observer, graph)
    assert collective.owners[captured_key].calls == 18
    assert collective.fallbacks[fallback_key] == 13
    assert all(value == 1 for key, value in collective.fallbacks.items() if key != fallback_key)
    assert observer.snapshot()["replayed_calls"] == 15


@pytest.mark.parametrize(
    "change",
    (
        "remove",
        "replace",
        "move",
        "counter-bool",
        "counter-negative",
        "counter-string",
        "fallback-missing",
        "fallback-bool",
    ),
)
def test_replay_rejects_touched_key_or_counter_mutation_without_partial_accounting(collective, change):
    observer, graph = collective.graph_observer, Graph()
    capture(collective, graph)
    owner_key = ("float16", 4096)
    owner = collective.owners[owner_key]
    other_owner = collective.owners["float32", 32]
    other_before = other_owner.calls
    observer.before_replay(graph)
    if change == "remove":
        collective.owners.pop(owner_key)
    elif change == "replace":
        collective.owners[owner_key] = SimpleNamespace(calls=owner.calls, state="active")
    elif change == "move":
        # The old O(N) identity search accepted a captured owner under the wrong
        # schema key. Keyed identity must reject this mapping mutation.
        collective.owners["float16", 8192] = collective.owners.pop(owner_key)
    elif change == "counter-bool":
        owner.calls = True
    elif change == "counter-negative":
        owner.calls = -1
    elif change == "counter-string":
        owner.calls = "8"
    elif change == "fallback-missing":
        collective.fallbacks.clear()
    else:
        collective.fallbacks["torch.bfloat16", 17, True] = True
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        observer.replayed(graph)
    assert other_owner.calls == other_before
    assert collective.state == owner.state == other_owner.state == "quarantined"
    assert observer.snapshot()["replay_submissions"] == observer.snapshot()["replayed_calls"] == 0
    assert graph.events == [] and graph.tasks


def test_zero_delta_capture_requires_no_geometry_or_fallback_accounting_on_replay(collective):
    graph, observer = Graph(), collective.graph_observer
    capture(collective, graph, (0, 0), 0)
    record = observer._records[id(graph)]
    assert record.calls == record.fallback_calls == ()
    assert len(record.retained_owners) == 2
    before = [owner.calls for owner in collective.owners.values()]
    replay(observer, graph)
    assert [owner.calls for owner in collective.owners.values()] == before
    assert observer.snapshot()["replay_submissions"] == 1
    assert observer.snapshot()["captured_calls"] == observer.snapshot()["replayed_calls"] == 0


@pytest.mark.parametrize("tracking", (False, True))
def test_capture_acquires_every_native_lease_before_entering_capture_scope(collective, tracking):
    observer = collective.graph_observer = CollectiveGraphObserver(collective, track_replays=tracking)
    graph = Graph()
    with observer.capture_scope(graph):
        assert all(len(owner.leases) == 1 for owner in collective.owners.values())
        assert observer.snapshot()["native_leases_acquired"] == 2
        assert observer.snapshot()["native_leases_released"] == 0
    observer.retire(lambda: None)
    assert all(not owner.leases for owner in collective.owners.values())
    assert observer.snapshot()["native_leases_acquired"] == observer.snapshot()["native_leases_released"] == 2


def test_native_leases_release_only_after_all_graph_resets_and_explicit_drain(collective):
    first, second, events = Graph(), Graph(), []
    observer = collective.graph_observer
    capture(collective, first)
    capture(collective, second)
    for index, owner in enumerate(collective.owners.values()):
        release = owner.release_graph_lease

        def checked_release(ticket, *, rank=index, original=release):
            assert events[0] == "drained"
            assert first.events == second.events == ["reset"]
            assert first.tasks == second.tasks == []
            events.append(("release", rank))
            original(ticket)

        owner.release_graph_lease = checked_release
    observer.retire(lambda: events.append("drained"))
    assert events == ["drained", ("release", 0), ("release", 1), ("release", 0), ("release", 1)]
    assert observer.snapshot()["native_leases_released"] == 4


@pytest.mark.parametrize("partial", (False, True))
def test_partial_native_lease_acquire_keeps_graph_owner_and_previous_tickets(collective, partial):
    owners, graph = tuple(collective.owners.values()), Graph()
    retained = object()

    def failed_acquire():
        if partial:
            owners[1].leases.append(retained)
        raise RuntimeError("native acquire failed with retained ticket")

    owners[1].acquire_graph_lease = failed_acquire
    observer = collective.graph_observer
    with pytest.raises(RuntimeError, match="native acquire failed"), observer.capture_scope(graph):
        pytest.fail("failed lease acquisition must not enter capture")
    assert len(owners[0].leases) == 1 and len(owners[1].leases) == int(partial)
    assert observer.snapshot()["native_leases_acquired"] == 1
    assert observer.snapshot()["native_leases_released"] == 0
    assert observer.snapshot()["failed_graphs"] == 1
    assert not graph.events and collective.state == "quarantined"
    assert observer._records[id(graph)].graph is graph


def test_null_lease_ticket_rejects_capture_without_claiming_a_lease(collective):
    owners = tuple(collective.owners.values())
    owners[0].acquire_graph_lease = lambda: None
    with pytest.raises(RuntimeError, match="null ticket"):
        collective.graph_observer.capture_begin(Graph())
    assert collective.graph_observer.snapshot()["native_leases_acquired"] == 0
    assert collective.state == "quarantined"


def test_reset_failure_keeps_every_native_lease_acquired(collective):
    observer, first, second = collective.graph_observer, Graph(), Graph()
    capture(collective, first)
    capture(collective, second)
    second.reset_error = RuntimeError("native reset failed")
    with pytest.raises(RuntimeError, match="native reset failed"):
        observer.retire(lambda: None)
    assert observer.snapshot()["native_leases_acquired"] == 4
    assert observer.snapshot()["native_leases_released"] == 0
    assert all(len(owner.leases) == 2 for owner in collective.owners.values())
    assert not observer.retired


def test_partial_lease_release_failure_retains_remaining_tickets_and_blocks_close_proof(collective):
    observer, first, second = collective.graph_observer, Graph(), Graph()
    capture(collective, first)
    capture(collective, second)
    owners = tuple(collective.owners.values())

    def failed_release(ticket):
        assert ticket in owners[1].leases
        raise RuntimeError("native release failed")

    owners[1].release_graph_lease = failed_release
    with pytest.raises(RuntimeError, match="native release failed"):
        observer.retire(lambda: None)
    assert first.events == second.events == ["reset"]
    assert observer.snapshot()["native_leases_acquired"] == 4
    assert observer.snapshot()["native_leases_released"] == 1
    assert [len(owner.leases) for owner in owners] == [1, 2]
    assert not observer.retired and collective.state == "quarantined"
    with pytest.raises(RuntimeError, match="quarantined"):
        observer.retire(lambda: pytest.fail("no retry after uncertain native release"))


def test_untracked_default_keeps_capture_only_semantics_and_rejects_diagnostic_callbacks(collective):
    observer = collective.graph_observer = CollectiveGraphObserver(collective)
    graph = Graph()
    capture(collective, graph)
    before = [owner.calls for owner in collective.owners.values()]
    for _ in range(10):
        graph.replay()
    assert [owner.calls for owner in collective.owners.values()] == before
    result = observer.snapshot()
    assert result["tracking_enabled"] is False and result["count_semantics"] == "capture-only"
    assert result["replay_submissions"] == result["replayed_calls"] == 0
    assert result["captured_calls"] == 4 and result["native_leases_acquired"] == 2
    with pytest.raises(RuntimeError, match="diagnostics are disabled"):
        observer.before_replay(graph)
    with pytest.raises(RuntimeError, match="diagnostics are disabled"):
        observer.replayed(graph)
    assert collective.state == "active" and not observer.retired


@pytest.mark.parametrize("tracking", (None, 0, 1, "false", [], {}))
def test_tracking_requires_exact_boolean(collective, tracking):
    with pytest.raises(ValueError, match="explicit bool"):
        CollectiveGraphObserver(collective, track_replays=tracking)


def test_capture_registration_strongly_retains_graph_and_partial_owner_after_failure(collective):
    observer, graph = collective.graph_observer, Graph()
    reference = weakref.ref(graph)
    error = KeyboardInterrupt("partial capture")
    with pytest.raises(KeyboardInterrupt) as caught, observer.capture_scope(graph):
        tuple(collective.owners.values())[0].calls += 1
        raise error
    assert caught.value is error and collective.state == "quarantined"
    assert all(owner.state == "quarantined" for owner in collective.owners.values())
    del graph
    gc.collect()
    assert reference() is not None and reference().events == []
    assert observer.snapshot()["failed_graphs"] == 1 and not observer.active_capture
    with pytest.raises(RuntimeError, match="quarantined"):
        observer.retire(lambda: pytest.fail("failed capture must retain without automatic cleanup"))


def test_retirement_stops_replay_before_drain_and_resets_all_graphs_before_metadata_clear(collective):
    observer, first, second, events = collective.graph_observer, Graph(), Graph(), []
    capture(collective, first)
    capture(collective, second)

    def synchronize():
        events.append("sync")
        with pytest.raises(RuntimeError, match="retiring"):
            observer.before_replay(first)
        assert first.tasks and second.tasks and not first.events and not second.events

    def reset_first():
        events.append("reset-first")
        assert first.tasks and second.tasks

    def reset_second():
        events.append("reset-second")
        assert first.tasks and second.tasks

    first.reset, second.reset = reset_first, reset_second
    result = observer.retire(synchronize)
    assert events == ["sync", "reset-first", "reset-second"]
    assert result["retired"] is True and result["reset_graphs"] == 2
    assert observer.retired and collective.state == "active"
    for graph in (first, second):
        assert graph.tasks == [] and graph.provider_sizes == graph.capture_resources == {}
    assert observer.retire(lambda: pytest.fail("retirement is idempotent")) == result
    with pytest.raises(RuntimeError, match="retired"):
        observer.before_replay(first)
    with pytest.raises(RuntimeError, match="retired"):
        observer.capture_begin(Graph())


@pytest.mark.parametrize("failure", ("drain", "first-reset", "second-reset"))
def test_retirement_failure_retains_every_graph_and_task_owner_without_retry(collective, failure):
    observer, first, second = collective.graph_observer, Graph(), Graph()
    capture(collective, first)
    capture(collective, second)
    error = RuntimeError("uncertain retirement")
    if failure == "first-reset":
        first.reset_error = error
    elif failure == "second-reset":
        second.reset_error = error

    def synchronize():
        if failure == "drain":
            raise error

    with pytest.raises(RuntimeError) as caught:
        observer.retire(synchronize)
    assert caught.value is error and collective.state == "quarantined" and not observer.retired
    assert first.tasks and second.tasks and observer.snapshot()["registered_graphs"] == 2
    assert first.events == ([] if failure == "drain" else ["reset"])
    assert second.events == (["reset"] if failure == "second-reset" else [])
    with pytest.raises(RuntimeError, match="quarantined"):
        observer.retire(lambda: pytest.fail("must not retry uncertain retirement"))


def test_raw_capture_unknown_graph_nested_capture_and_double_replay_are_rejected(collective):
    observer, graph = collective.graph_observer, Graph()
    with pytest.raises(RuntimeError, match="registered"):
        observer.require_capture()
    with pytest.raises(RuntimeError, match="registered"):
        observer.before_replay(graph)
    observer.capture_begin(graph)
    with pytest.raises(RuntimeError, match="nested"):
        observer.capture_begin(Graph())
    with pytest.raises(RuntimeError, match="registered"):
        observer.capture_end(Graph())
    with pytest.raises(RuntimeError, match="active"):
        observer.retire(lambda: pytest.fail("cannot drain within capture"))
    observer.capture_end(graph)
    with pytest.raises(RuntimeError, match="repeated"):
        observer.capture_begin(graph)
    with pytest.raises(RuntimeError, match="not admitted"):
        observer.replayed(graph)
    observer.before_replay(graph)
    with pytest.raises(RuntimeError, match="pending"):
        observer.before_replay(graph)
    with pytest.raises(RuntimeError, match="pending"):
        observer.retire(lambda: pytest.fail("cannot drain within replay admission"))
    observer.replayed(graph)
    with pytest.raises(RuntimeError, match="not admitted"):
        observer.replayed(graph)


def test_invalid_capture_or_retirement_interfaces_fail_before_native_calls(collective):
    observer = collective.graph_observer
    with pytest.raises(ValueError, match="reset and replay"):
        observer.capture_begin(object())
    with pytest.raises(ValueError, match="drain callable"):
        observer.retire(True)
    assert observer.snapshot()["registered_graphs"] == 0


@pytest.mark.parametrize(
    "change", ("owners", "owner-identity", "counter-decrease", "counter-bool", "fallback-decrease", "fallback-missing")
)
def test_capture_counter_or_owner_drift_quarantines_without_dropping_graph(collective, change):
    observer, graph = collective.graph_observer, Graph()
    observer.capture_begin(graph)
    owner = tuple(collective.owners.values())[0]
    if change == "owners":
        collective.owners.pop(("float32", 32))
    elif change == "owner-identity":
        collective.owners["float16", 4096] = SimpleNamespace(calls=5, state="active")
    elif change == "counter-decrease":
        owner.calls -= 1
    elif change == "counter-bool":
        owner.calls = True
    elif change == "fallback-decrease":
        collective.fallbacks["torch.bfloat16", 17, True] = 1
    else:
        collective.fallbacks.clear()
    with pytest.raises((RuntimeError, ValueError)):
        observer.capture_end(graph)
    assert collective.state == "quarantined" and observer.snapshot()["failed_graphs"] == 1
    assert graph.tasks and not graph.events


@pytest.mark.parametrize("failure", ("validation", "replay"))
def test_replay_validation_or_native_failure_does_not_credit_counters(collective, failure):
    observer, graph = collective.graph_observer, Graph()
    capture(collective, graph)
    before = [owner.calls for owner in collective.owners.values()]
    error = RuntimeError("wrong stream or device")
    if failure == "validation":
        collective.validation_error = error
        with pytest.raises(RuntimeError) as caught:
            observer.before_replay(graph)
        assert graph.events == []
    else:
        graph.replay_error = error
        observer.before_replay(graph)
        with pytest.raises(RuntimeError) as caught:
            try:
                graph.replay()
            except BaseException as raised:
                observer.replay_failed(graph, raised)
                raise
    assert caught.value is error and collective.state == "quarantined"
    assert [owner.calls for owner in collective.owners.values()] == before
    assert observer.snapshot()["replay_submissions"] == 0 and graph.tasks


@pytest.fixture
def wrapper_fixture(collective):
    class Mode(Enum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2

    class Tensor:
        pass

    state = SimpleNamespace(
        mode=Mode.FULL,
        events=[],
        updatable=False,
        imports=[],
        runnable_error=None,
        forward=SimpleNamespace(
            batch_descriptor="decode", cudagraph_runtime_mode=Mode.FULL, capturing=False, attn_metadata={}
        ),
        observer=collective.graph_observer,
    )

    @contextmanager
    def graph_context(graph, pool):
        state.events.append("capture-enter")
        if state.observer is not None:
            assert state.observer.require_capture() is graph
        try:
            yield
        finally:
            state.events.append("capture-exit")

    def import_dependency(name, globals=None, locals=None, fromlist=(), level=0):
        state.imports.append(name)
        if name == "vllm.distributed":
            return SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(device_communicator=SimpleNamespace(custom_sum=collective))
            )
        if name == "vllm.model_executor.offloader.base":
            return SimpleNamespace(
                get_offloader=lambda: SimpleNamespace(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
            )
        raise AssertionError("unexpected wrapper dependency: " + name)

    torch = SimpleNamespace(
        Tensor=Tensor,
        npu=SimpleNamespace(
            NPUGraph=Graph,
            Stream=object,
            graph=graph_context,
            current_stream=lambda: SimpleNamespace(synchronize=lambda: state.events.append("sync-current")),
        ),
    )
    namespace = dict(
        __name__="isolated_acl_graph",
        __builtins__={**vars(builtins), "__import__": import_dependency},
        dataclasses=dataclasses,
        dataclass=dataclasses.dataclass,
        torch=torch,
        Callable=Callable,
        Any=Any,
        VllmConfig=object,
        BatchDescriptor=object,
        CUDAGraphOptions=object,
        CUDAGraphMode=Mode,
        UpdatableGraph=Graph,
        ExitStack=ExitStack,
        get_forward_context=lambda: state.forward,
        validate_cudagraph_capturing_enabled=lambda: None,
        super_kernel_scope=lambda *args: nullcontext(),
        weak_ref_tensors=lambda value: value,
        weak_ref_workspaces=lambda value: None,
        compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
        _graph_params=None,
        _draft_graph_params=None,
        _draft_graph_prefill_params=None,
        _EXTRA_CTX=SimpleNamespace(is_draft_model=False),
        use_updatable_graph=lambda backend: state.updatable,
        ContextSource=lambda value: value,
        SharedSource=lambda value: value,
        logger=SimpleNamespace(info_once=lambda *args: None),
        _is_old_hdk_capture_error=lambda error: False,
        _is_stream_resource_capture_error=lambda error: False,
        partial=partial,
    )
    source = ROOT / "vllm_ascend/compilation/acl_graph.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    parsed.body = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name in ("ACLGraphEntry", "ACLGraphWrapper")
    ]
    exec(compile(parsed, str(source), "exec"), namespace)
    wrapper = object.__new__(namespace["ACLGraphWrapper"])

    def runnable():
        if state.runnable_error:
            raise state.runnable_error
        for owner in collective.owners.values():
            owner.calls += 1
        return "output"

    wrapper.__dict__.update(
        runnable=runnable,
        vllm_config=SimpleNamespace(additional_config={"npu_collective": {"execution_mode": "graph"}}),
        runtime_mode=Mode.FULL,
        concrete_aclgraph_entries={},
        is_debugging_mode=False,
        aclgraph_options=SimpleNamespace(debug_log_enable=False, gc_disable=False, weak_ref_output=False),
        graph_pool=None,
        enable_super_kernel=False,
        enable_enpu=False,
        use_eagle=False,
        attn_backend=None,
        update_stream=object(),
        draft_model_metadata=[],
        _collective_graph_observer=None,
    )
    state.wrapper, state.collective = wrapper, collective
    return state


@pytest.mark.parametrize("updatable,enpu", ((False, False), (True, False), (True, True)))
def test_actual_wrapper_registers_capture_and_counts_each_replay_branch(wrapper_fixture, updatable, enpu):
    state, wrapper = wrapper_fixture, wrapper_fixture.wrapper
    state.updatable, wrapper.enable_enpu = updatable, enpu
    assert wrapper() == "output"
    graph = wrapper.concrete_aclgraph_entries["decode"].aclgraph
    assert state.observer.snapshot()["captured_graphs"] == 1
    assert state.events == ["capture-enter", "capture-exit"]
    before = [owner.calls for owner in state.collective.owners.values()]
    assert wrapper() == "output"
    assert [owner.calls for owner in state.collective.owners.values()] == [value + 1 for value in before]
    expected = ["update", "replay"] if enpu else (["replay", "update"] if updatable else ["replay"])
    assert graph.events == expected
    assert state.observer.snapshot()["replay_submissions"] == 1
    state.observer.retire(lambda: None)
    with pytest.raises(RuntimeError, match="retired"):
        wrapper()
    assert graph.events == expected + ["reset"]


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_actual_wrapper_retains_failed_capture_before_cache_assignment(wrapper_fixture, error_type):
    state = wrapper_fixture
    state.runnable_error = error_type("capture failed")
    with pytest.raises(error_type, match="capture failed"):
        state.wrapper()
    assert state.wrapper.concrete_aclgraph_entries["decode"].aclgraph is None
    assert state.observer.snapshot()["registered_graphs"] == state.observer.snapshot()["failed_graphs"] == 1
    assert state.collective.state == "quarantined"


def test_actual_wrapper_accounts_replay_before_later_attention_update_failure(wrapper_fixture):
    state = wrapper_fixture
    state.updatable = True
    state.wrapper()
    graph = state.wrapper.concrete_aclgraph_entries["decode"].aclgraph
    graph.update_error = RuntimeError("attention update failed after replay")
    with pytest.raises(RuntimeError, match="attention update failed"):
        state.wrapper()
    assert graph.events == ["replay", "update"]
    assert state.observer.snapshot()["replay_submissions"] == 1


def test_actual_wrapper_propagates_replay_failure_without_credit_or_cleanup(wrapper_fixture):
    state = wrapper_fixture
    state.wrapper()
    graph = state.wrapper.concrete_aclgraph_entries["decode"].aclgraph
    graph.replay_error = RuntimeError("partial graph enqueue")
    with pytest.raises(RuntimeError, match="partial graph enqueue"):
        state.wrapper()
    assert state.observer.snapshot()["replay_submissions"] == 0
    assert state.observer.snapshot()["failed_graphs"] == 1
    assert graph.events == ["replay"] and graph.tasks


@pytest.mark.parametrize("custom", (None, {}, {"execution_mode": "eager"}))
def test_default_wrapper_never_imports_optional_collective_or_changes_replay(wrapper_fixture, custom):
    state = wrapper_fixture
    state.wrapper.vllm_config.additional_config = {} if custom is None else {"npu_collective": custom}
    state.observer = None
    assert state.wrapper() == "output"
    graph = state.wrapper.concrete_aclgraph_entries["decode"].aclgraph
    assert state.wrapper() == "output" and graph.events == ["replay"]
    assert state.wrapper._collective_graph_observer is None
    assert "vllm.distributed" not in state.imports


def test_graph_mode_requires_observer_before_native_capture(wrapper_fixture):
    state = wrapper_fixture
    state.collective.graph_observer = None
    with pytest.raises(RuntimeError, match="explicit lifecycle observer"):
        state.wrapper()
    assert state.events == []


def test_eager_warmup_bypasses_capture_observation(wrapper_fixture):
    state = wrapper_fixture
    state.forward.cudagraph_runtime_mode = type(state.forward.cudagraph_runtime_mode).NONE
    assert state.wrapper() == "output"
    assert state.observer.snapshot()["registered_graphs"] == 0
    assert not state.imports and not state.events


@pytest.mark.parametrize("updatable,enpu", ((False, False), (True, False), (True, True)))
def test_production_wrapper_caches_native_replay_without_observer_callbacks(wrapper_fixture, updatable, enpu):
    state = wrapper_fixture
    observer = state.collective.graph_observer = CollectiveGraphObserver(state.collective)
    state.observer = observer
    state.updatable, state.wrapper.enable_enpu = updatable, enpu
    assert state.wrapper() == "output"
    entry = state.wrapper.concrete_aclgraph_entries["decode"]
    assert entry.replay_callable.__self__ is entry.aclgraph
    assert entry.replay_callable.__func__ is Graph.replay

    def forbidden(*args):
        pytest.fail("production graph replay invoked Python observer diagnostics")

    observer.before_replay = observer.replayed = observer.replay_failed = forbidden
    state.collective.validate_graph_replay = forbidden
    before = [owner.calls for owner in state.collective.owners.values()]
    for _ in range(10):
        assert state.wrapper() == "output"
    assert [owner.calls for owner in state.collective.owners.values()] == before
    assert observer.snapshot()["replay_submissions"] == observer.snapshot()["replayed_calls"] == 0
    assert observer.snapshot()["native_leases_acquired"] == 2
    observer.retire(lambda: None)
    assert observer.snapshot()["native_leases_released"] == 2
    # The serialized shutdown RPC clears wrapper entries after native reset;
    # a subsequent call cannot recapture against the retired collective.
    state.wrapper.concrete_aclgraph_entries.clear()
    with pytest.raises(RuntimeError, match="retired"):
        state.wrapper()


@pytest.mark.parametrize("updatable,enpu", ((False, False), (True, False), (True, True)))
def test_untracked_replay_or_update_exception_quarantines_only_on_error_path(wrapper_fixture, updatable, enpu):
    state = wrapper_fixture
    observer = state.collective.graph_observer = CollectiveGraphObserver(state.collective)
    state.observer = observer
    state.updatable, state.wrapper.enable_enpu = updatable, enpu
    state.wrapper()
    graph = state.wrapper.concrete_aclgraph_entries["decode"].aclgraph
    if updatable:
        graph.update_error = RuntimeError("unobserved native update failed")
    else:
        graph.replay_error = RuntimeError("unobserved native replay failed")
    with pytest.raises(RuntimeError, match="unobserved native"):
        state.wrapper()
    assert state.collective.state == "quarantined"
    assert observer.snapshot()["failed_graphs"] == 1
    assert observer.snapshot()["replay_submissions"] == observer.snapshot()["replayed_calls"] == 0
    assert observer.snapshot()["native_leases_acquired"] == 2
    assert observer.snapshot()["native_leases_released"] == 0
