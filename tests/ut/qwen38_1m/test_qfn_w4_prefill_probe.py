# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.qwen4exp import benchmark_route_finalize_310 as benchmark
from tools.qwen4exp.resident_candidates import qfn_w4_prefill_probe as probe


def setup(monkeypatch):
    npu = SimpleNamespace(
        mem_get_info=lambda: (2**32, 2**35), is_current_stream_capturing=lambda: False, synchronize=lambda: None
    )
    torch = SimpleNamespace(npu=npu, inference_mode=contextlib.nullcontext)
    worker = SimpleNamespace(model_runner=SimpleNamespace(execute_model_state=None))
    monkeypatch.setattr(benchmark, "verify_build", lambda path, **kwargs: {})

    def load(path, *, load_library, owner, expected_version):
        assert load_library
        assert expected_version in (1, 2, 3)
        owner["finalize"] = object()
        return owner

    monkeypatch.setattr(benchmark, "load_resources", load)
    monkeypatch.setattr(probe, "run_validation", lambda *args: {"passed": True})
    return worker, torch


def request(action, rows=(650,), *, version=1, build="/tmp/private-build"):
    return probe.parse_request(json.dumps({"action": action, "build": build, "rows": list(rows), "version": version}))


def test_preflight_then_load_then_validate_and_timing(monkeypatch):
    worker, torch = setup(monkeypatch)
    with pytest.raises(RuntimeError, match="preflight"):
        probe.run_action(worker, request("load"), torch)
    assert probe.run_action(worker, request("preflight"), torch)["passed"]
    assert probe.run_action(worker, request("load"), torch)["passed"]
    state = getattr(worker, probe.STATE_ATTRIBUTE)
    resource = state["resources"]["finalize"]
    assert probe.run_action(worker, request("load"), torch)["already_loaded"]
    with pytest.raises(RuntimeError, match="validate every"):
        probe.run_action(worker, request("timing"), torch)
    assert probe.run_action(worker, request("validate"), torch)["passed"]
    assert probe.run_action(worker, request("timing"), torch)["passed"]
    assert state["resources"]["finalize"] is resource and not state["failed"]


def test_partial_load_retains_instance_owner(monkeypatch):
    worker, torch = setup(monkeypatch)
    probe.run_action(worker, request("preflight"), torch)
    resource = object()

    def load(path, *, load_library, owner, expected_version):
        owner["partial"] = resource
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(benchmark, "load_resources", load)
    with pytest.raises(RuntimeError, match="constructor failed"):
        probe.run_action(worker, request("load"), torch)
    state = getattr(worker, probe.STATE_ATTRIBUTE)
    assert state["failed"] and state["resources"]["partial"] is resource
    with pytest.raises(RuntimeError, match="retain resources"):
        probe.run_action(worker, request("load"), torch)


def test_failed_numerical_validation_retains_resources(monkeypatch):
    worker, torch = setup(monkeypatch)
    probe.run_action(worker, request("preflight"), torch)
    probe.run_action(worker, request("load"), torch)
    monkeypatch.setattr(probe, "run_validation", lambda *args: {"passed": False})
    assert not probe.run_action(worker, request("validate"), torch)["passed"]
    state = getattr(worker, probe.STATE_ATTRIBUTE)
    assert state["failed"] and state["resources"]["finalize"] is not None


@pytest.mark.parametrize("kind", ["pending", "profile", "cpu_profile", "capture", "memory", "capacity"])
def test_admission_refuses_unsafe_gate(monkeypatch, kind):
    worker, torch = setup(monkeypatch)
    probe.run_action(worker, request("preflight"), torch)
    if kind == "pending":
        worker.model_runner.execute_model_state = object()
    elif kind == "profile":
        worker.profiler = object()
    elif kind == "cpu_profile":
        worker._qfn_cpu_profile = object()
    elif kind == "capture":
        torch.npu.is_current_stream_capturing = lambda: True
    elif kind == "memory":
        torch.npu.mem_get_info = lambda: (0, 2**35)
        assert not probe.run_action(worker, request("load"), torch)["passed"]
        assert not getattr(worker, probe.STATE_ATTRIBUTE)["resources"]
        return
    with pytest.raises(RuntimeError):
        probe.run_action(worker, request("load", (6400,) if kind == "capacity" else (650,)), torch)
    assert not getattr(worker, probe.STATE_ATTRIBUTE)["resources"]


@pytest.mark.parametrize(
    "updates",
    [
        {"rows": [641]},
        {"rows": [650, 650]},
        {"rows": [True]},
        {"repeats": 6},
        {"build": "relative"},
        {"projection": "other"},
        {"action": "delete"},
        {"unexpected": 1},
        {"version": 0},
        {"version": 4},
        {"version": True},
        {"version": "2"},
    ],
)
def test_reject_invalid_rpc_contract(updates):
    payload = {"action": "preflight", "build": "/tmp/native", **updates}
    with pytest.raises(ValueError):
        probe.parse_request(json.dumps(payload))


def test_rpc_target_and_original_argument_delegation():
    calls = []
    rpc = probe.make_rpc(lambda self, output=None: calls.append(output))
    worker = SimpleNamespace(rank=3)
    rpc(worker, "/tmp/cpu-profile")
    assert calls == ["/tmp/cpu-profile"]
    assert probe.TARGET.endswith("NPUWorker310.qfn_cpu_profile_stop")
    receipt = rpc(worker, '{"action":"invalid"}')
    assert receipt["rank"] == 3 and not receipt["passed"] and "ValueError" in receipt["error"]


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("mutation", [None, "namespace", "binary", "source", "path", "header"])
def test_verify_build_integrity_before_native_constructor(monkeypatch, tmp_path, mutation, version):
    payload = b"immutable test artifact"
    digest = hashlib.sha256(payload).hexdigest()
    sources = {"qualified.h": digest, "kernel.cpp": digest}
    if version >= 2:
        sources.update({benchmark.OWNED_BRIDGE_SOURCE: digest, benchmark.OWNED_ARGUMENTS_SOURCE: digest})
    for name in sources:
        (tmp_path / name).write_bytes(payload)
    monkeypatch.setattr(benchmark, "QUALIFIED_HEADERS", {"qualified.h": digest})
    artifacts = {}
    for name in ("bridge", "native_route_finalize", "native_prefill_schedule_probe"):
        path = tmp_path / name
        path.write_bytes(payload)
        artifacts[name] = {"path": str(path), "sha256": digest}
    provenance = {
        "namespace": f"qwen_route_finalize_v{version}",
        "sources": sources,
        "bridge": artifacts.pop("bridge"),
        "binaries": artifacts,
    }
    if version >= 2:
        provenance["bridge_source"] = benchmark.OWNED_BRIDGE_SOURCE
    if version == 3:
        provenance["flush_host_queue"] = True
    if mutation == "namespace":
        provenance["namespace"] = "other"
    elif mutation == "binary":
        (tmp_path / "native_route_finalize").write_text("changed")
    elif mutation == "source":
        (tmp_path / "kernel.cpp").write_text("changed")
    elif mutation == "path":
        provenance["bridge"]["path"] = "relative.so"
    elif mutation == "header":
        (tmp_path / "qualified.h").write_text("changed")
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    if mutation is None:
        assert benchmark.verify_build(tmp_path) == provenance
        assert benchmark.verify_build(tmp_path, expected_version=version) == provenance
        with pytest.raises(ValueError, match="requested version"):
            benchmark.verify_build(tmp_path, expected_version=2 if version == 1 else 1)
    else:
        with pytest.raises(ValueError):
            benchmark.verify_build(tmp_path)


def test_legacy_request_defaults_to_v1():
    value = probe.parse_request(json.dumps(dict(action="preflight", build="/tmp/private")))
    assert value["version"] == 1
    assert probe.state_attribute(1) == probe.STATE_ATTRIBUTE
    assert probe.state_attribute(2) == probe.STATE_ATTRIBUTE_V2
    assert probe.state_attribute(3) == probe.STATE_ATTRIBUTE_V3


def test_v2_load_leaves_failed_v1_resources_and_traceback_untouched(monkeypatch):
    worker, torch = setup(monkeypatch)
    old_failure = RuntimeError("v1 native failure")
    old_resources = {"retained": object()}
    old_traceback = object()
    old_state = dict(
        build="/old/v1",
        version=1,
        resources=old_resources,
        failed=True,
        failure=old_failure,
        failure_traceback=old_traceback,
    )
    setattr(worker, probe.STATE_ATTRIBUTE, old_state)
    for action in ("preflight", "load", "validate", "timing"):
        assert probe.run_action(worker, request(action, version=2), torch)["passed"]
    assert getattr(worker, probe.STATE_ATTRIBUTE) is old_state
    assert old_state["resources"] is old_resources
    assert old_state["failure"] is old_failure and old_state["failure_traceback"] is old_traceback
    new_state = getattr(worker, probe.STATE_ATTRIBUTE_V2)
    assert new_state["resources"] is not old_resources and new_state["loaded"]
    with pytest.raises(RuntimeError, match="do not replace or unload"):
        probe.run_action(worker, request("preflight", version=2, build="/new/v2"), torch)
    assert getattr(worker, probe.STATE_ATTRIBUTE_V2) is new_state


def test_v2_failure_retention_is_separate_and_retry_keeps_original_traceback(monkeypatch):
    worker, torch = setup(monkeypatch)
    for version in (1, 2):
        for action in ("preflight", "load"):
            probe.run_action(worker, request(action, version=version), torch)
    v1 = getattr(worker, probe.STATE_ATTRIBUTE)
    v1_resource = v1["resources"]["finalize"]

    def fail(*args):
        raise RuntimeError("v2 uncertain submission")

    monkeypatch.setattr(probe, "run_validation", fail)
    with pytest.raises(RuntimeError, match="v2 uncertain"):
        probe.run_action(worker, request("validate", version=2), torch)
    v2 = getattr(worker, probe.STATE_ATTRIBUTE_V2)
    failure, traceback = v2["failure"], v2["failure_traceback"]
    with pytest.raises(RuntimeError, match="retain resources"):
        probe.run_action(worker, request("validate", version=2), torch)
    assert v2["failure"] is failure and v2["failure_traceback"] is traceback
    assert not v1["failed"] and "failure_traceback" not in v1
    assert v1["resources"]["finalize"] is v1_resource


def test_preflight_version_is_forwarded_and_mismatch_does_not_admit_load(monkeypatch):
    worker, torch = setup(monkeypatch)

    def verify(path, *, expected_version):
        assert expected_version == 2
        raise ValueError("namespace differs from requested version")

    monkeypatch.setattr(benchmark, "verify_build", verify)
    with pytest.raises(ValueError, match="requested version"):
        probe.run_action(worker, request("preflight", version=2), torch)
    state = getattr(worker, probe.STATE_ATTRIBUTE_V2)
    assert state["preflight"] is None and not state["resources"]


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("mutation", ["missing_bridge", "missing_header", "wrong_selected_bridge"])
def test_owned_versions_require_explicit_bridge_and_frozen_sources(tmp_path, mutation, version):
    provenance = dict(
        namespace=f"qwen_route_finalize_v{version}",
        flush_host_queue=version == 3,
        bridge_source=benchmark.OWNED_BRIDGE_SOURCE,
        sources={benchmark.OWNED_BRIDGE_SOURCE: "hash", benchmark.OWNED_ARGUMENTS_SOURCE: "hash"},
    )
    if mutation == "missing_bridge":
        del provenance["sources"][benchmark.OWNED_BRIDGE_SOURCE]
    elif mutation == "missing_header":
        del provenance["sources"][benchmark.OWNED_ARGUMENTS_SOURCE]
    else:
        provenance["bridge_source"] = "reconstruction_bridge.cpp"
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="explicitly selected frozen owned bridge"):
        benchmark.verify_build(tmp_path)


@pytest.mark.parametrize("version", [2, 3])
def test_resource_namespace_changes_but_kernel_symbols_remain_v1(monkeypatch, version):
    created = []
    provenance = dict(
        namespace=f"qwen_route_finalize_v{version}",
        bridge=dict(path="/v2/bridge.so"),
        binaries={
            "native_route_finalize": dict(path="/v2/finalize.bin"),
            "native_prefill_schedule_probe": dict(path="/v2/schedule.bin"),
        },
    )

    def constructor(path, symbol):
        created.append((path, symbol))
        return (path, symbol)

    launch = object()
    monkeypatch.setattr(benchmark, "verify_build", lambda path, **kwargs: provenance)
    monkeypatch.setattr(
        benchmark,
        "torch",
        SimpleNamespace(
            classes=SimpleNamespace(**{provenance["namespace"]: SimpleNamespace(Kernel=constructor)}),
            ops=SimpleNamespace(
                **{provenance["namespace"]: SimpleNamespace(launch=launch)},
                load_library=lambda path: created.append(path),
            ),
        ),
    )
    monkeypatch.setattr(benchmark, "RouteFinalizer", lambda resource, function: (resource, function))
    monkeypatch.setattr(benchmark, "QualifiedSchedule", lambda resource, function: (resource, function))
    owner = {}
    assert benchmark.load_resources("/v2", load_library=True, owner=owner, expected_version=version) is owner
    assert created == [
        "/v2/bridge.so",
        ("/v2/finalize.bin", "qwen_route_finalize_v1"),
        *[("/v2/schedule.bin", f"qwen_prefill_{name}_v1") for name in ("baseline", "small", "local_count")],
    ]
    assert all(resource[1] is launch for name, resource in owner.items() if name != "eager_only")
    if version == 3:
        assert owner["eager_only"] is True
    else:
        assert "eager_only" not in owner


@pytest.mark.parametrize("version,flag", [(1, True), (2, True), (3, False), (3, None), (3, 1), (3, "true")])
def test_host_queue_flush_provenance_is_explicit_and_exclusive_to_v3(tmp_path, version, flag):
    provenance = dict(namespace=f"qwen_route_finalize_v{version}")
    if flag is not None:
        provenance["flush_host_queue"] = flag
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="flush_host_queue"):
        benchmark.verify_build(tmp_path, expected_version=version)


def test_v3_failure_does_not_fallback_to_or_modify_either_old_version(monkeypatch):
    worker, torch = setup(monkeypatch)
    prior = []
    for version in (1, 2):
        for action in ("preflight", "load"):
            probe.run_action(worker, request(action, version=version), torch)
        state = getattr(worker, probe.state_attribute(version))
        state.update(failed=True, failure=RuntimeError(f"old v{version} error"), failure_traceback=object())
        prior.append((state, dict(state)))
    for action in ("preflight", "load"):
        assert probe.run_action(worker, request(action, version=3), torch)["passed"]
    state = getattr(worker, probe.STATE_ATTRIBUTE_V3)
    active_resources = []

    def validation(worker, resources, request, torch):
        assert request["version"] == 3
        active_resources.append(resources)
        raise RuntimeError("v3 native queue failure")

    monkeypatch.setattr(probe, "run_validation", validation)
    with pytest.raises(RuntimeError, match="v3 native queue failure"):
        probe.run_action(worker, request("validate", version=3), torch)
    first_failure, first_traceback = state["failure"], state["failure_traceback"]
    with pytest.raises(RuntimeError, match="retain resources"):
        probe.run_action(worker, request("validate", version=3), torch)
    assert active_resources == [state["resources"]]
    assert state["failure"] is first_failure and state["failure_traceback"] is first_traceback
    for version, (old, snapshot) in enumerate(prior, 1):
        assert getattr(worker, probe.state_attribute(version)) is old
        assert old == snapshot
        assert old["resources"] is snapshot["resources"]
    with pytest.raises(RuntimeError, match="do not replace or unload"):
        probe.run_action(worker, request("preflight", version=3, build="/replacement/v3"), torch)
    assert getattr(worker, probe.STATE_ATTRIBUTE_V3) is state
