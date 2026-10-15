# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import qfn_collective_worker as module

ROOT = Path(__file__).resolve().parents[3]


def owner(dtype="float16", elements=4096, calls=7):
    return SimpleNamespace(
        schedule=SimpleNamespace(dtype=dtype, elements=elements, identity="b" * 64),
        calls=calls,
        plan="secret native pointer",
        allocations="secret device addresses",
        edges="secret IPC keys",
    )


def facade():
    return SimpleNamespace(
        members=(7, 3, 9, 2),
        rank=9,
        identity="a" * 64,
        state="active",
        owners={("float16", 4096): owner(), ("float32", 32): owner("float32", 32, 2)},
        fallbacks={("torch.bfloat16", 4096, True): 3, ("float16", 17, False): 1, ("dict", None, None): 4},
        config={"acl_library": "/secret/library/path", "private": "secret"},
        session="secret session",
        claimed_stream="secret stream pointer",
        pid=123456,
    )


@pytest.fixture
def worker(monkeypatch):
    current = SimpleNamespace(custom_sum=facade())
    calls = []

    def close():
        calls.append("close")
        current.custom_sum.state = "closed"

    current.close_custom_collective = close
    monkeypatch.setattr(module, "_communicator", lambda: current)
    return module.QfnCollectiveWorkerExtension(), current, calls


def test_module_import_does_not_import_torch_vllm_or_query_a_device(monkeypatch):
    imports = []
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        imports.append(name)
        if name.startswith(("torch", "vllm")):
            pytest.fail("worker extension imported accelerator dependencies at module startup")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = ROOT / "tools/sidecard/qfn_collective_worker.py"
    spec = importlib.util.spec_from_file_location("isolated_qfn_worker", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert callable(loaded.QfnCollectiveWorkerExtension().npu_collective_status)
    assert not any(name.startswith(("torch", "vllm")) for name in imports)


def test_real_tp_group_lookup_is_lazy_and_reads_device_communicator_property(monkeypatch):
    imports, groups = [], []
    communicator = SimpleNamespace(custom_sum=None)

    class Group:
        @property
        def device_communicator(self):
            groups.append("device-communicator")
            return communicator

    def get_tp_group():
        groups.append("get-tp-group")
        return Group()

    original = builtins.__import__

    def lazy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.distributed":
            imports.append((name, fromlist, level))
            return SimpleNamespace(get_tp_group=get_tp_group)
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", lazy)
    extension = module.QfnCollectiveWorkerExtension()
    assert imports == groups == []
    assert extension.npu_collective_status()["state"] == "disabled"
    assert imports == [("vllm.distributed", ("get_tp_group",), 0)]
    assert groups == ["get-tp-group", "device-communicator"]
    assert extension.npu_collective_close()["state"] == "disabled"
    assert len(imports) == 2 and groups == ["get-tp-group", "device-communicator"] * 2


@pytest.mark.parametrize("missing_attribute", (False, True))
def test_baseline_status_and_close_need_no_custom_or_close_method(worker, missing_attribute):
    extension, communicator, calls = worker
    if missing_attribute:
        del communicator.custom_sum
    else:
        communicator.custom_sum = None
    del communicator.close_custom_collective
    expected = dict(schema_version=1, enabled=False, state="disabled", geometries=[], fallback_schemas=[])
    assert extension.npu_collective_status() == expected
    assert extension.npu_collective_close() == expected
    assert not calls


def test_status_reports_only_sanitized_json_metadata_without_mutable_aliases(worker):
    extension, communicator, calls = worker
    result = extension.npu_collective_status()
    assert result == dict(
        schema_version=1,
        enabled=True,
        state="active",
        rank=9,
        members=[7, 3, 9, 2],
        config_sha256="a" * 64,
        geometries=[
            dict(dtype="float16", elements=4096, schedule_sha256="b" * 64, calls=7),
            dict(dtype="float32", elements=32, schedule_sha256="b" * 64, calls=2),
        ],
        fallback_schemas=[
            dict(dtype="torch.bfloat16", elements=4096, contiguous=True, calls=3),
            dict(dtype="float16", elements=17, contiguous=False, calls=1),
            dict(dtype="dict", elements=None, contiguous=None, calls=4),
        ],
    )
    assert json.loads(json.dumps(result)) == result
    assert "secret" not in json.dumps(result) and not calls
    result["members"].append(99)
    result["geometries"][0]["calls"] = 999
    result["fallback_schemas"][0]["calls"] = 999
    assert communicator.custom_sum.members == (7, 3, 9, 2)
    assert communicator.custom_sum.owners["float16", 4096].calls == 7
    assert communicator.custom_sum.fallbacks["torch.bfloat16", 4096, True] == 3


def test_unknown_fallback_type_never_exports_arbitrary_strings_or_object_repr(worker):
    extension, communicator, _ = worker
    communicator.custom_sum.fallbacks = {("secret/path/0x123456", 0, False): 1}
    result = extension.npu_collective_status()
    assert result["fallback_schemas"] == [dict(dtype="unsupported", elements=0, contiguous=False, calls=1)]
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("state", ("new", "active", "closed", "quarantined"))
def test_status_observes_metadata_in_all_retained_facade_states_without_cleanup(worker, state):
    extension, communicator, calls = worker
    communicator.custom_sum.state = state
    assert extension.npu_collective_status()["state"] == state
    assert not calls


def test_close_calls_communicator_then_returns_closed_status_without_engine_shutdown(worker):
    extension, communicator, calls = worker
    result = extension.npu_collective_close()
    assert calls == ["close"] and communicator.custom_sum.state == "closed"
    assert result["enabled"] is True and result["state"] == "closed"
    assert result["geometries"][0]["calls"] == 7


def test_close_delegates_repeated_calls_to_communicators_idempotent_protocol(worker):
    extension, communicator, calls = worker

    def idempotent_close():
        if communicator.custom_sum.state != "closed":
            calls.append("released")
            communicator.custom_sum.state = "closed"

    communicator.close_custom_collective = idempotent_close
    assert extension.npu_collective_close() == extension.npu_collective_close()
    assert calls == ["released"]


@pytest.mark.parametrize("exception_type", (RuntimeError, KeyboardInterrupt))
def test_close_failure_propagates_with_same_retained_owners_and_no_fallback(worker, exception_type):
    extension, communicator, calls = worker
    retained = communicator.custom_sum
    error = exception_type("uncertain shutdown")

    def failed_close():
        calls.append("attempted")
        retained.state = "quarantined"
        raise error

    communicator.close_custom_collective = failed_close
    with pytest.raises(exception_type, match="uncertain shutdown") as caught:
        extension.npu_collective_close()
    assert caught.value is error and communicator.custom_sum is retained and retained.owners
    assert calls == ["attempted"]
    assert extension.npu_collective_status()["state"] == "quarantined"


def test_close_without_closed_ownership_proof_is_an_error(worker):
    extension, communicator, calls = worker
    communicator.close_custom_collective = lambda: calls.append("incomplete")
    with pytest.raises(RuntimeError, match="did not establish closed ownership"):
        extension.npu_collective_close()
    assert calls == ["incomplete"] and communicator.custom_sum.state == "active"


@pytest.mark.parametrize(
    "attribute,value",
    (
        ("members", (7, 7)),
        ("members", (True, 3, 9, 2)),
        ("members", ()),
        ("members", "secret"),
        ("rank", True),
        ("rank", 99),
        ("identity", "secret"),
        ("identity", "A" * 64),
        ("state", "secret/path"),
        ("state", True),
    ),
)
def test_invalid_identity_state_or_members_fail_closed_without_exporting_strings(worker, attribute, value):
    extension, communicator, calls = worker
    setattr(communicator.custom_sum, attribute, value)
    with pytest.raises(ValueError):
        extension.npu_collective_status()
    assert not calls


@pytest.mark.parametrize("calls", (True, -1, "7", 1.5))
def test_geometry_counters_reject_coercions(worker, calls):
    extension, communicator, _ = worker
    communicator.custom_sum.owners["float16", 4096].calls = calls
    with pytest.raises(ValueError, match="integer counters"):
        extension.npu_collective_status()


@pytest.mark.parametrize(
    "key", (("float16", True), ("float16", 0), ("secret", 4096), ("float16", 32), ("float16", 4096, 1))
)
def test_invalid_or_mismatched_fixed_geometry_is_rejected(worker, key):
    extension, communicator, _ = worker
    communicator.custom_sum.owners = {key: owner()}
    with pytest.raises(ValueError, match="geometry"):
        extension.npu_collective_status()


@pytest.mark.parametrize(
    "key,calls",
    (
        (("float16", True, True), 1),
        (("float16", -1, True), 1),
        (("float16", 4096, 1), 1),
        (("float16", 4096), 1),
        ((1, 4096, True), 1),
        (("float16", 4096, True), True),
        (("float16", 4096, True), -1),
    ),
)
def test_fallback_geometry_and_counter_types_are_exact(worker, key, calls):
    extension, communicator, _ = worker
    communicator.custom_sum.fallbacks = {key: calls}
    with pytest.raises(ValueError):
        extension.npu_collective_status()


def graph_observer(calls):
    snapshot = dict(
        registered_graphs=2,
        captured_graphs=2,
        failed_graphs=0,
        replay_submissions=5,
        captured_calls=10,
        replayed_calls=50,
        reset_graphs=0,
        active_capture=False,
        retired=False,
        lifecycle_only=True,
        tracking_enabled=True,
        count_semantics="capture-and-replay-submissions",
        native_leases_acquired=4,
        native_leases_released=0,
    )

    def retire(synchronize):
        if not snapshot["retired"]:
            synchronize()
            calls.append("reset")
            snapshot.update(retired=True, reset_graphs=2, native_leases_released=4)

    return SimpleNamespace(snapshot=lambda: snapshot, retire=retire), snapshot


def install_graph_observer(worker, monkeypatch):
    extension, communicator, calls = worker
    observer, snapshot = graph_observer(calls)
    communicator.custom_sum.graph_observer = observer
    communicator.custom_sum.device_index = 2
    communicator.custom_sum.torch = SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda device: calls.append(("sync", device)))
    )
    monkeypatch.setattr(
        module,
        "_clear_graph_wrapper_caches",
        lambda value: calls.append("clear") if value is observer else pytest.fail("foreign observer"),
    )
    return extension, communicator, calls, snapshot


def test_graph_status_attaches_only_copied_sanitized_observer_scalars(worker, monkeypatch):
    extension, _, calls, snapshot = install_graph_observer(worker, monkeypatch)
    result = extension.npu_collective_status()
    assert result["graph_observer"] == snapshot and not calls
    result["graph_observer"]["replay_submissions"] = 999
    assert snapshot["replay_submissions"] == 5
    assert "secret" not in json.dumps(result)


def test_production_graph_status_marks_capture_only_and_keeps_replay_counters_unobserved(worker, monkeypatch):
    extension, _, calls, snapshot = install_graph_observer(worker, monkeypatch)
    snapshot.update(tracking_enabled=False, count_semantics="capture-only", replay_submissions=0, replayed_calls=0)
    result = extension.npu_collective_status()["graph_observer"]
    assert result["tracking_enabled"] is False and result["count_semantics"] == "capture-only"
    assert result["captured_calls"] == 10 and result["native_leases_acquired"] == 4
    assert not calls


@pytest.mark.parametrize(
    "field,bad",
    (
        ("replay_submissions", True),
        ("reset_graphs", -1),
        ("active_capture", 0),
        ("lifecycle_only", False),
        ("IPC_keys", "secret"),
    ),
)
def test_graph_status_rejects_coercions_unknown_fields_and_nonsanitized_values(worker, monkeypatch, field, bad):
    extension, _, _, snapshot = install_graph_observer(worker, monkeypatch)
    snapshot[field] = bad
    with pytest.raises(ValueError):
        extension.npu_collective_status()


def test_graph_close_synchronizes_retires_and_clears_before_native_release(worker, monkeypatch):
    extension, communicator, calls, snapshot = install_graph_observer(worker, monkeypatch)
    result = extension.npu_collective_close()
    assert calls == [("sync", 2), "reset", "clear", "close"]
    assert communicator.custom_sum.state == "closed" and snapshot["retired"] is True
    assert result["graph_observer"]["reset_graphs"] == 2


@pytest.mark.parametrize("fault", ("sync", "reset", "clear"))
def test_graph_retirement_failure_retains_native_owners_and_prevents_close(worker, monkeypatch, fault):
    extension, communicator, calls, _ = install_graph_observer(worker, monkeypatch)

    def fail(*args):
        raise RuntimeError("uncertain graph retirement")

    if fault == "sync":
        communicator.custom_sum.torch.npu.synchronize = fail
    elif fault == "reset":
        communicator.custom_sum.graph_observer.retire = lambda synchronize: (synchronize(), fail())
    else:
        monkeypatch.setattr(module, "_clear_graph_wrapper_caches", fail)
    with pytest.raises(RuntimeError, match="retirement"):
        extension.npu_collective_close()
    assert "close" not in calls
    assert communicator.custom_sum.owners and communicator.custom_sum.state == "active"


@pytest.mark.parametrize("path", ("runner", "drafter", "encoder"))
def test_unmanaged_runner_graphs_cannot_be_silently_claimed_retired(worker, monkeypatch, path):
    extension, communicator, calls, _ = install_graph_observer(worker, monkeypatch)
    manager = SimpleNamespace(graphs={"live": object()})
    if path == "runner":
        extension.model_runner = SimpleNamespace(cudagraph_manager=manager)
    elif path == "encoder":
        extension.model_runner = SimpleNamespace(encoder_cudagraph_manager=manager)
    else:
        extension.model_runner = SimpleNamespace(drafter=SimpleNamespace(cudagraph_manager=manager))
    with pytest.raises(RuntimeError, match="unmanaged"):
        extension.npu_collective_close()
    assert not calls and communicator.custom_sum.state == "active"


def test_wrapper_cache_cleanup_is_limited_to_this_retired_observer(monkeypatch):
    observer = object()
    first = SimpleNamespace(
        _collective_graph_observer=observer, concrete_aclgraph_entries={"graph": object()}, first_run_finished=True
    )
    foreign = SimpleNamespace(
        _collective_graph_observer=object(), concrete_aclgraph_entries={"other": object()}, first_run_finished=True
    )
    original = builtins.__import__

    def lazy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_ascend.compilation":
            return SimpleNamespace(acl_graph=SimpleNamespace(_acl_graph_wrappers=[first, foreign]))
        if name == "vllm.compilation.breakable_cudagraph":
            return SimpleNamespace(BreakableCUDAGraphWrapper=SimpleNamespace(_all_instances=[]))
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", lazy)
    module._clear_graph_wrapper_caches(observer)
    assert first.concrete_aclgraph_entries == {} and first.first_run_finished is False
    assert foreign.concrete_aclgraph_entries and foreign.first_run_finished is True


def test_breakable_wrapper_cache_cleanup_uses_owned_observer_after_retirement(worker, monkeypatch):
    cleanup = module._clear_graph_wrapper_caches
    extension, communicator, calls, snapshot = install_graph_observer(worker, monkeypatch)
    observer = communicator.custom_sum.graph_observer

    class Breakable:
        def __init__(self, retained_observer):
            self._collective_graph_observer = retained_observer
            self.entries = {"native segmented graph": object()}

        def clear_graphs(self):
            assert snapshot["retired"] is True
            assert "close" not in calls
            calls.append("clear-breakable")
            self.entries.clear()

    owned, foreign = Breakable(observer), Breakable(object())

    class Forwarded:
        # Selection must read actual wrapper ownership, never a forwarded
        # runnable attribute that happens to refer to the current observer.
        entries = {"foreign graph": object()}

        def __getattr__(self, name):
            if name == "_collective_graph_observer":
                return observer
            raise AttributeError(name)

        def clear_graphs(self):
            pytest.fail("cleanup followed a forwarded observer attribute")

    forwarded = Forwarded()
    original = builtins.__import__

    def lazy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_ascend.compilation":
            return SimpleNamespace(acl_graph=SimpleNamespace(_acl_graph_wrappers=[]))
        if name == "vllm.compilation.breakable_cudagraph":
            return SimpleNamespace(
                BreakableCUDAGraphWrapper=SimpleNamespace(_all_instances=[owned, foreign, forwarded])
            )
        return original(name, globals, locals, fromlist, level)

    # Restore the real helper after the generic graph-worker fixture replaced
    # it, while injecting only the two accelerator framework registries.
    monkeypatch.setattr(module, "_clear_graph_wrapper_caches", cleanup)
    monkeypatch.setattr(builtins, "__import__", lazy)
    result = extension.npu_collective_close()
    assert calls == [("sync", 2), "reset", "clear-breakable", "close"]
    assert not owned.entries and foreign.entries and forwarded.entries
    assert result["state"] == "closed"
