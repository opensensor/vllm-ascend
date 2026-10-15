# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host dispatch and native-lease ownership gates; no device execution."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qsa_graph_dispatch as candidate
from tools.sidecard.npu_collective_graph import CollectiveGraphObserver


def metadata(length=512, queries=(3,)):
    return SimpleNamespace(
        seq_lens_cpu=torch.tensor([length] * len(queries), dtype=torch.int32),
        query_lens_cpu=torch.tensor(queries, dtype=torch.int32),
        num_actual_tokens=sum(queries),
        num_prefills=0,
        num_decodes=len(queries),
        attn_state=SimpleNamespace(name="SpecDecoding"),
        block_tables=torch.zeros(len(queries), 4096, dtype=torch.int32),
        query_start_loc=torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32),
        slot_mapping=torch.zeros(sum(queries), dtype=torch.int64),
        seq_lens=torch.tensor([length] * len(queries), dtype=torch.int32),
    )


class Graph:
    def __init__(self):
        self.resets = 0
        self.capture_token = None
        self.tasks = ["owned"]
        self.error = None

    def replay(self):
        return "replayed"

    def reset(self):
        self.resets += 1
        if self.error:
            raise self.error


class Owner:
    def __init__(self):
        self.state, self.calls = "active", 0
        self.tickets = set()
        self.error = None

    def acquire_graph_lease(self):
        ticket = object()
        self.tickets.add(ticket)
        return ticket

    def release_graph_lease(self, ticket):
        if self.error:
            raise self.error
        self.tickets.remove(ticket)


def observer():
    owners = {index: Owner() for index in range(2)}
    facade = SimpleNamespace(state="active", owners=owners, fallbacks={}, _context=lambda: None)
    return CollectiveGraphObserver(facade, track_replays=False)


def register(owner, graph):
    with owner.capture_scope(graph):
        for lease_owner in owner.collective.owners.values():
            lease_owner.calls += 1


def setup_dispatch():
    boundaries = metadata()
    context = SimpleNamespace(batch_descriptor="three", attn_metadata={"qsa0": boundaries, "qsa1": boundaries})
    geometries = dict.fromkeys(context.attn_metadata, candidate.PrefixGeometry())
    dispatcher = candidate.PrefixDispatcher(geometries)
    wrapper = SimpleNamespace(entries={})
    events = []

    def invoke():
        entry = wrapper.entries.setdefault(
            context.batch_descriptor, SimpleNamespace(capture=None, output=None, input_addresses=None)
        )
        if entry.capture is None:
            entry.capture = Graph()
            entry.prefix = bool(getattr(context, "_qfn_qsa_prefix_capture", None))
            events.append(("capture", entry.prefix))
        else:
            events.append(("replay", entry.prefix))
        return "model output"

    def retire(graph):
        events.append(("retire", graph))
        graph.reset()
        return {"native_leases_released": 2}

    return boundaries, context, dispatcher, wrapper, events, invoke, retire


def test_prefix_capture_replay_then_general_replacement_stays_one_entry():
    boundaries, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    assert dispatch.call(wrapper, context, (), {}, invoke, retire) == "model output"
    first = wrapper.entries["three"].capture
    for length in (0, 29, 512, 2048, 2051):
        boundaries.seq_lens_cpu.fill_(length)
        dispatch.call(wrapper, context, (), {}, invoke, retire)
    assert first.resets == 0
    assert all(prefix for action, prefix in events if action in ("capture", "replay"))
    boundaries.seq_lens_cpu.fill_(2052)
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    assert first.resets == 1
    assert len(wrapper.entries) == 1
    assert events[-2:] == [("capture", False), ("replay", False)]
    for length in (262144, 512):
        boundaries.seq_lens_cpu.fill_(length)
        dispatch.call(wrapper, context, (), {}, invoke, retire)
        assert events[-1] == ("replay", False)
    assert len(dispatch.retirement_receipts) == 1
    assert not hasattr(context, "_qfn_qsa_prefix_capture")


def test_scheduler_metadata_object_can_change_with_same_buffers_and_new_lengths():
    boundaries, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    current = SimpleNamespace(**vars(boundaries))
    current.seq_lens_cpu = torch.tensor([999], dtype=torch.int32)
    context.attn_metadata = {name: current for name in context.attn_metadata}
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    assert events[-1] == ("replay", True)


@pytest.mark.parametrize("name", ["block_tables", "query_start_loc", "slot_mapping", "seq_lens"])
def test_changed_device_pointer_fails_before_replay(name):
    boundaries, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    setattr(boundaries, name, getattr(boundaries, name).clone())
    with pytest.raises(RuntimeError, match="pause and reset all ranks"):
        dispatch.call(wrapper, context, (), {}, invoke, retire)
    assert len(events) == 1
    assert wrapper.entries["three"].capture.resets == 0


def test_draft_step_with_different_model_input_pointer_cannot_replay_other_step():
    _, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    first, second = torch.zeros(3, 16), torch.zeros(3, 16)
    dispatch.call(wrapper, context, (), {"inputs_embeds": first}, invoke, retire)
    with pytest.raises(RuntimeError, match="addresses"):
        dispatch.call(wrapper, context, (), {"inputs_embeds": second}, invoke, retire)
    assert len(events) == 1


def test_changed_packed_query_layout_rejected_even_with_same_token_count():
    boundaries, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    boundaries = metadata(512, queries=(3, 3))
    context.attn_metadata = {name: boundaries for name in context.attn_metadata}
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    boundaries.query_lens_cpu.copy_(torch.tensor([1, 5]))
    with pytest.raises(RuntimeError, match="packed geometry"):
        dispatch.call(wrapper, context, (), {}, invoke, retire)
    assert len(events) == 1


def test_unowned_existing_capture_is_not_adopted():
    _, context, dispatch, wrapper, _, invoke, retire = setup_dispatch()
    wrapper.entries["three"] = SimpleNamespace(capture=Graph())
    with pytest.raises(RuntimeError, match="retiring existing"):
        dispatch.call(wrapper, context, (), {}, invoke, retire)


def test_capture_monitor_window_wraps_retirement_and_new_capture_only():
    boundaries, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    allowed = [False]

    @contextmanager
    def window():
        events.append(("window", "enter"))
        allowed[0] = True
        try:
            yield
        finally:
            allowed[0] = False
            events.append(("window", "leave"))

    def validate():
        if not allowed[0]:
            raise RuntimeError("capture disabled")

    dispatch.call(wrapper, context, (), {}, invoke, retire)
    boundaries.seq_lens_cpu.fill_(2052)
    with pytest.raises(RuntimeError, match="capture disabled"):
        dispatch.call(wrapper, context, (), {}, invoke, retire, prepare_recapture=validate)
    assert not dispatch.failed and len(events) == 1
    dispatch.call(wrapper, context, (), {}, invoke, retire, prepare_recapture=validate, recapture_window=window)
    assert events[-5][0] == "window" and events[-4][0] == "retire"
    assert events[-3:-1] == [("capture", False), ("replay", False)]
    assert events[-1] == ("window", "leave") and not allowed[0]


def test_failure_preserves_context_attribute_and_prevents_automatic_retry():
    boundaries, context, dispatch, wrapper, _, invoke, retire = setup_dispatch()
    context._qfn_qsa_prefix_capture = "outer"
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    boundaries.seq_lens_cpu.fill_(2052)

    def broken():
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        dispatch.call(wrapper, context, (), {}, broken, retire)
    assert context._qfn_qsa_prefix_capture == "outer"
    with pytest.raises(RuntimeError, match="coordinated reset"):
        dispatch.call(wrapper, context, (), {}, invoke, retire)


def test_single_capture_retirement_preserves_other_native_owners_and_allows_global_retire():
    owner, first, second = observer(), Graph(), Graph()
    register(owner, first)
    register(owner, second)
    events = []
    receipt = candidate.retire_capture(owner, first, lambda: events.append("drain"))
    assert events == ["drain"]
    assert receipt == {"captures_reset": 1, "native_leases_released": 2, "other_captures": 1}
    assert first.resets == 1 and second.resets == 0
    assert all(len(lease_owner.tickets) == 1 for lease_owner in owner.collective.owners.values())
    assert owner.collective.state == "active" and not owner.retired
    assert first.tasks == []
    third = Graph()
    register(owner, third)
    owner.retire(lambda: events.append("final drain"))
    assert all(not lease_owner.tickets for lease_owner in owner.collective.owners.values())
    assert first.resets == 1 and second.resets == third.resets == 1


@pytest.mark.parametrize("failure", ["drain", "token", "reset", "release"])
def test_retirement_failure_retains_record_and_remaining_tickets_and_quarantines(failure):
    owner, graph = observer(), Graph()
    register(owner, graph)

    def drain():
        if failure == "drain":
            raise RuntimeError("drain failed")

    if failure == "token":
        graph.capture_token = object()
    if failure == "reset":
        graph.error = RuntimeError("reset failed")
    if failure == "release":
        list(owner.collective.owners.values())[-1].error = RuntimeError("release failed")
    with pytest.raises(RuntimeError):
        candidate.retire_capture(owner, graph, drain)
    assert id(graph) in owner._records
    assert owner.collective.state == "quarantined"
    assert all(item.state == "quarantined" for item in owner.collective.owners.values())
    assert sum(len(item.tickets) for item in owner.collective.owners.values()) >= 1
    with pytest.raises(RuntimeError, match="quarantined"):
        candidate.retire_capture(owner, graph, lambda: None)


def test_active_capture_or_tracking_blocks_retirement_before_drain():
    owner, graph = observer(), Graph()
    register(owner, graph)
    owner.track_replays = True
    with pytest.raises(RuntimeError, match="unobserved"):
        candidate.retire_capture(owner, graph, lambda: pytest.fail("drained"))
    owner.track_replays = False
    owner.capture_begin(Graph())
    with pytest.raises(RuntimeError, match="active"):
        candidate.retire_capture(owner, graph, lambda: pytest.fail("drained"))


def test_causal_prefix_requires_explicit_geometry_and_ignores_rope_coordinates():
    context = SimpleNamespace(_qfn_qsa_prefix_capture={"qsa": candidate.PrefixGeometry()})
    logical = torch.tensor([2048, 2049, 2050], dtype=torch.int32)
    output = candidate.prefix_selection_for_capture(context, "qsa", logical, compress_ratio=4, token_topk=2048)
    assert torch.equal(output.group_counts, torch.tensor([512, 512, 512]))
    assert torch.equal(output.tail_counts, torch.tensor([1, 2, 3]))
    with pytest.raises(RuntimeError, match="not admitted"):
        candidate.prefix_selection_for_capture(context, "different", logical, compress_ratio=4, token_topk=2048)
    with pytest.raises(ValueError, match="MRoPE"):
        candidate.prefix_selection_for_capture(context, "qsa", logical.expand(3, -1), compress_ratio=4, token_topk=2048)


def test_installer_builds_three_replacements_without_modifying_live_methods(monkeypatch):
    from vllm_ascend.models.qwen4_exp import model

    original = model._QSAAttention.forward
    monkeypatch.setattr(candidate, "RESIDENT_RECAPTURE_WINDOW_QUALIFIED", True)
    replacements = candidate.replacements()
    assert len(replacements) == 3
    assert model._QSAAttention.forward is original
    assert replacements["vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward"] is not original
    assert replacements["vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward"].__globals__ is original.__globals__
    monkeypatch.setattr(candidate.inspect, "getsource", lambda function: "def forward(self):\n    return None\n")
    with pytest.raises(RuntimeError, match="source seams"):
        candidate.replacements()


def test_hot_replay_does_not_mutate_capture_contract_or_retirement_records():
    boundaries, context, dispatch, wrapper, _, invoke, retire = setup_dispatch()
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    contract = dispatch.contracts["three"]
    for length in (29, 30, 31):
        boundaries.seq_lens_cpu.fill_(length)
        dispatch.call(wrapper, context, (), {}, invoke, retire)
        assert dispatch.contracts["three"] is contract
        assert dispatch.retirement_receipts == []
        assert not hasattr(context, "_qfn_qsa_prefix_capture")


def test_capture_monitor_restores_disabled_setting_after_exception(monkeypatch):
    from vllm.compilation import monitor

    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", False)
    dispatcher, _, _ = stream_window_setup(monkeypatch)
    with pytest.raises(RuntimeError, match="failed capture"), candidate.serialized_recapture_window(dispatcher):
        assert monitor.cudagraph_capturing_enabled
        raise RuntimeError("failed capture")
    assert not monitor.cudagraph_capturing_enabled


def test_payload_can_be_prepared_again_while_candidate_is_installed(monkeypatch):
    from vllm_ascend.models.qwen4_exp import model

    target = "vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward"
    monkeypatch.setattr(candidate, "RESIDENT_RECAPTURE_WINDOW_QUALIFIED", True)
    first = candidate.replacements()[target]
    original = model._QSAAttention.forward
    monkeypatch.setattr(model._QSAAttention, "forward", first)
    second = candidate.replacements()[target]
    assert second._qfn_qsa_base_forward is original


def generation_wrapper():
    return SimpleNamespace(
        entries={},
        runnable=object(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=262144),
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
    )


def test_parent_resident_renewal_rebuilds_failed_dispatcher_and_sticky_general_state(monkeypatch):
    from tools.sidecard.qfn_resident_lifecycle import renew_collective_graphs

    old = observer()
    facade = old.collective
    facade.graph_observer = old
    wrapper = generation_wrapper()
    wrapper._collective_graph_observer = old
    monkeypatch.setattr(candidate, "discover_geometries", lambda root, context: {"qsa": candidate.PrefixGeometry()})
    first = candidate.dispatcher_for_observer(wrapper, old)
    graph = Graph()
    register(old, graph)
    wrapper.entries["three"] = SimpleNamespace(capture=graph)
    first.contracts["three"] = candidate.CapturedContract(graph, "general", ())
    first.failed = True  # Host candidate failure; collective itself remains valid.
    first.retirement_receipts.append({"native_leases_released": 14})
    owners = facade.owners
    retirement = renew_collective_graphs(
        facade,
        [wrapper],
        synchronize=lambda: None,
        clear_state=lambda wrappers: [item.entries.clear() for item in wrappers],
        observer_factory=CollectiveGraphObserver,
    )
    assert retirement["retired"]
    fresh = candidate.dispatcher_for_observer(wrapper, facade.graph_observer)
    assert fresh is not first and fresh.observer is facade.graph_observer
    assert not fresh.failed and fresh.contracts == {} and fresh.retirement_receipts == []
    assert facade.owners is owners and all(not item.tickets for item in owners.values())
    assert wrapper._qfn_qsa_prefix_dispatch is fresh


def test_same_observer_keeps_dispatcher_state_without_cold_rediscovery(monkeypatch):
    owner, wrapper = observer(), generation_wrapper()
    discoveries = []

    def discover(root, context):
        discoveries.append(context)
        return {"qsa": candidate.PrefixGeometry()}

    monkeypatch.setattr(candidate, "discover_geometries", discover)
    first = candidate.dispatcher_for_observer(wrapper, owner)
    first.failed = True
    first.contracts["three"] = "general capture contract"
    assert candidate.dispatcher_for_observer(wrapper, owner) is first
    assert first.failed and first.contracts["three"] == "general capture contract"
    assert discoveries == [262144]


@pytest.mark.parametrize("clear_entries,retire_old", [(True, False), (False, True), (False, False)])
def test_new_observer_rejects_unfinished_retirement_or_uncleared_entries(monkeypatch, clear_entries, retire_old):
    old, wrapper = observer(), generation_wrapper()
    monkeypatch.setattr(candidate, "discover_geometries", lambda root, context: {"qsa": candidate.PrefixGeometry()})
    prior = candidate.dispatcher_for_observer(wrapper, old)
    graph = Graph()
    register(old, graph)
    wrapper.entries["three"] = SimpleNamespace(capture=graph)
    if retire_old:
        old.retire(lambda: None)
    if clear_entries:
        wrapper.entries.clear()
    fresh = CollectiveGraphObserver(old.collective, track_replays=False)
    with pytest.raises(RuntimeError, match="retired graph caches|native lease retirement"):
        candidate.dispatcher_for_observer(wrapper, fresh)
    assert wrapper._qfn_qsa_prefix_dispatch is prior


def test_native_failure_cannot_be_hidden_by_swapping_observers(monkeypatch):
    old, wrapper = observer(), generation_wrapper()
    monkeypatch.setattr(candidate, "discover_geometries", lambda root, context: {"qsa": candidate.PrefixGeometry()})
    prior = candidate.dispatcher_for_observer(wrapper, old)
    graph = Graph()
    register(old, graph)
    graph.error = RuntimeError("native reset failed")
    with pytest.raises(RuntimeError):
        old.retire(lambda: None)
    fresh = CollectiveGraphObserver(old.collective, track_replays=False)
    with pytest.raises(RuntimeError, match="native lease retirement"):
        candidate.dispatcher_for_observer(wrapper, fresh)
    assert wrapper._qfn_qsa_prefix_dispatch is prior
    assert all(item.tickets for item in old.collective.owners.values())


def test_nonqsa_wrapper_skip_is_cached_per_observer_generation(monkeypatch):
    owner, wrapper = observer(), generation_wrapper()
    calls = []
    monkeypatch.setattr(candidate, "discover_geometries", lambda root, context: calls.append(context) or {})
    assert candidate.dispatcher_for_observer(wrapper, owner) is None
    wrapper.entries["foreign"] = SimpleNamespace(capture=Graph())
    assert candidate.dispatcher_for_observer(wrapper, owner) is None
    assert calls == [262144]


def test_unbound_older_candidate_requires_explicit_controller_clear(monkeypatch):
    owner, wrapper = observer(), generation_wrapper()
    wrapper._qfn_qsa_prefix_dispatch = candidate.PrefixDispatcher({"qsa": candidate.PrefixGeometry()})
    with pytest.raises(RuntimeError, match="legacy QSA dispatcher"):
        candidate.dispatcher_for_observer(wrapper, owner)


def test_full_source_payload_loads_through_actual_resident_loader(monkeypatch):
    from tools.glm_perf import resident_control

    # The source executes in a ModuleType absent from sys.modules and inherits
    # resident_control's future annotations. Exercise that real loader, while
    # substituting only the unavailable NPU wrapper target during resolution.
    # CPU loader seam only; this does not qualify the resident hardware gate.
    source = (
        Path(candidate.__file__)
        .read_text()
        .replace("RESIDENT_RECAPTURE_WINDOW_QUALIFIED = False", "RESIDENT_RECAPTURE_WINDOW_QUALIFIED = True")
    )
    wrapper_class = type("BreakableACLGraphWrapper", (), {"__call__": lambda self: None})
    wrapper_module = SimpleNamespace(BreakableACLGraphWrapper=wrapper_class)
    worker_class = type("NPUWorker310", (), {"qfn_graph_inventory": lambda self: []})
    original_import = resident_control.importlib.import_module

    def import_target(name, *args, **kwargs):
        if name == "vllm_ascend.compilation.breakable_aclgraph":
            return wrapper_module
        if name == "vllm_ascend._310p.worker_310p":
            return SimpleNamespace(NPUWorker310=worker_class)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(resident_control.importlib, "import_module", import_target)
    control = resident_control.Control(generation="a" * 32, candidate="qsa_graph_dispatch", source=source)
    result = resident_control.prepare_replacements(control)
    assert len(result) == 3
    assert {replacement.attribute for replacement in result} == {"forward", "__call__", "qfn_graph_inventory"}
    wrapper = next(replacement.function for replacement in result if replacement.attribute == "__call__")
    geometry_type = wrapper.__globals__["PrefixGeometry"]
    assert geometry_type().max_prefix_length == 2051
    with pytest.raises(AttributeError):
        geometry_type().token_topk = 4096


def test_target_wrapper_excludes_draft_and_unknown_models():
    for name in (*candidate.TARGET_MODEL_CLASSES, "Qwen4ExpMTP", "UnknownModel"):
        root = type(name, (), {"modules": lambda self: ()})()
        assert candidate.is_target_wrapper(SimpleNamespace(runnable=root)) == (name in candidate.TARGET_MODEL_CLASSES)
        assert candidate.is_target_wrapper(SimpleNamespace(runnable=root.modules)) == (
            name in candidate.TARGET_MODEL_CLASSES
        )


def test_target_callback_inventory_rejects_unknown_or_miscounted_eager_work():
    graph = type("NPUGraph", (), {"replay": lambda self: None})()
    graph.replay.__func__.__name__ = "replay"

    def scratch():
        pass

    scratch.__module__ = "vllm_ascend.models.qwen4_exp.model"
    capture = SimpleNamespace(segments=[graph.replay, scratch], num_graphs=1, num_eager_breaks=1)
    for name in candidate.SCRATCH_CALLBACKS:
        scratch.__qualname__ = name
        candidate.validate_target_capture(capture)
    scratch.__qualname__ = "dispatch_routed_experts"
    with pytest.raises(RuntimeError, match="unreviewed"):
        candidate.validate_target_capture(capture)
    scratch.__qualname__ = next(iter(candidate.SCRATCH_CALLBACKS))
    capture.num_eager_breaks = 2
    with pytest.raises(RuntimeError, match="inconsistent"):
        candidate.validate_target_capture(capture)


def test_boundary_runs_device_graph_once_and_scratch_twice():
    boundaries, context, dispatch, wrapper, _, _, retire = setup_dispatch()
    state = {"graph": 0, "scratch": 0}

    def invoke():
        entry = wrapper.entries.setdefault("three", SimpleNamespace(capture=None, output=None, input_addresses=None))
        state["scratch"] += 1
        if entry.capture is None:
            entry.capture = Graph()  # Hardware: recording does not execute graph ops.
        else:
            state["graph"] += 1
        return dict(state)

    assert dispatch.call(wrapper, context, (), {}, invoke, retire) == {"graph": 0, "scratch": 1}
    boundaries.seq_lens_cpu.fill_(2052)
    assert dispatch.call(wrapper, context, (), {}, invoke, retire) == {"graph": 1, "scratch": 3}
    assert dispatch.call(wrapper, context, (), {}, invoke, retire) == {"graph": 2, "scratch": 4}


def test_boundary_unknown_callback_rejected_before_old_capture_retirement():
    boundaries, context, dispatch, wrapper, events, invoke, retire = setup_dispatch()
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    boundaries.seq_lens_cpu.fill_(2052)

    def reject(capture):
        raise RuntimeError("unreviewed callback")

    with pytest.raises(RuntimeError, match="unreviewed"):
        dispatch.call(wrapper, context, (), {}, invoke, retire, validate_capture=reject)
    assert len(events) == 1 and not dispatch.failed
    assert wrapper.entries["three"].capture.resets == 0


def test_boundary_replay_failure_disallows_retry():
    boundaries, context, dispatch, wrapper, _, invoke, retire = setup_dispatch()
    dispatch.call(wrapper, context, (), {}, invoke, retire)
    boundaries.seq_lens_cpu.fill_(2052)
    calls = []

    def invoke_then_fail():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("new replay failed")
        return invoke()

    with pytest.raises(RuntimeError, match="new replay failed"):
        dispatch.call(wrapper, context, (), {}, invoke_then_fail, retire)
    assert dispatch.failed and len(calls) == 2


@pytest.mark.parametrize("field", ["events", "handles", "attn_params", "workspaces"])
def test_global_attention_ownership_blocks_isolated_replacement(field):
    parameters = SimpleNamespace(events={3: []}, handles={3: []}, attn_params={3: []}, workspaces={3: None})
    candidate.validate_empty_graph_parameters(None)
    candidate.validate_empty_graph_parameters(parameters)
    getattr(parameters, field)[3] = torch.empty(1, device="meta") if field == "workspaces" else [object()]
    with pytest.raises(RuntimeError, match="global attention"):
        candidate.validate_empty_graph_parameters(parameters)


def stream_window_setup(monkeypatch):
    events, streams, fences = [], [], []

    class Stream:
        def __init__(self):
            self.name = len(streams)
            streams.append(self)

        def wait_event(self, event):
            events.append(("wait", self.name, event.name))

    class Event:
        def __init__(self):
            self.name = len(fences)
            fences.append(self)

        def record(self, stream):
            events.append(("record", stream.name, self.name))

    default = Stream()
    current = [default]

    @contextmanager
    def use_stream(stream):
        previous = current[0]
        events.append(("enter", stream.name))
        current[0] = stream
        try:
            yield
        finally:
            events.append(("exit", stream.name))
            current[0] = previous

    npu = SimpleNamespace(
        Stream=Stream,
        Event=Event,
        current_stream=lambda: current[0],
        default_stream=lambda: default,
        stream=use_stream,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    dispatcher = candidate.PrefixDispatcher({"qsa": candidate.PrefixGeometry()})
    return dispatcher, events, (streams, fences, current)


@pytest.mark.parametrize("previous_monitor", [False, True])
def test_exact_recapture_window_orders_nondefault_stream_and_reuses_resources(monkeypatch, previous_monitor):
    from vllm.compilation import monitor

    dispatcher, events, resources = stream_window_setup(monkeypatch)
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", previous_monitor)
    for _ in range(2):
        with candidate.serialized_recapture_window(dispatcher):
            assert torch.npu.current_stream() != torch.npu.default_stream()
            assert monitor.cudagraph_capturing_enabled
            events.append(("capture_then_replay",))
        assert monitor.cudagraph_capturing_enabled == previous_monitor
        assert torch.npu.current_stream() == torch.npu.default_stream()
    expected = [
        ("record", 0, 0),
        ("wait", 1, 0),
        ("enter", 1),
        ("capture_then_replay",),
        ("record", 1, 1),
        ("exit", 1),
        ("wait", 0, 1),
    ]
    assert events == expected * 2
    assert len(resources[0]) == len(resources[1]) == 2
    assert dispatcher.recapture_resources == (resources[0][1], *resources[1])


def test_window_failure_retains_stream_and_events_without_consumer_submission(monkeypatch):
    dispatcher, events, resources = stream_window_setup(monkeypatch)
    with pytest.raises(RuntimeError, match="capture failed"), candidate.serialized_recapture_window(dispatcher):
        raise RuntimeError("capture failed")
    assert dispatcher.failed
    assert dispatcher.recapture_resources == (resources[0][1], *resources[1])
    assert events == [("record", 0, 0), ("wait", 1, 0), ("enter", 1), ("exit", 1)]
    assert torch.npu.current_stream() == torch.npu.default_stream()
    with pytest.raises(RuntimeError, match="coordinated retirement"), candidate.serialized_recapture_window(dispatcher):
        pytest.fail("retried failed capture")
    assert len(events) == 4


def test_default_stream_cannot_be_used_as_recapture_stream(monkeypatch):
    dispatcher, events, _ = stream_window_setup(monkeypatch)
    dispatcher.recapture_resources = (torch.npu.default_stream(), torch.npu.Event(), torch.npu.Event())
    with pytest.raises(RuntimeError, match="distinct non-default"), candidate.serialized_recapture_window(dispatcher):
        pytest.fail("entered capture")
    assert events == []


def test_default_factory_is_blocked_before_source_or_hardware_access(monkeypatch):
    assert candidate.RESIDENT_RECAPTURE_WINDOW_QUALIFIED is False
    monkeypatch.setattr(candidate.inspect, "getsource", lambda fn: pytest.fail("source accessed"))
    with pytest.raises(RuntimeError, match="exact resident recapture window requires hardware qualification"):
        candidate.replacements()
