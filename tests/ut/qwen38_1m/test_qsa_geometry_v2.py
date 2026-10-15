# SPDX-License-Identifier: Apache-2.0
"""Host contract/failure gates. NPU graph parity remains a private hardware gate."""

import hashlib
import importlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import qsa_geometry_v2 as ops
from tools.qwen4exp.resident_candidates import qfn_qsa_geometry_v2 as binding


def test_compiled_shared_integer_helper_matches_wrapping_floor(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for actual shared integer helper")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "check.cpp"
    source.write_text(
        '#include <iostream>\n#include "qsa_geometry_integer_v2.h"\n'
        "int main() { int64_t p; while (std::cin >> p) { "
        "auto g=qwen_geometry_v2::Position(static_cast<int64_t>(p)); "
        'std::cout << g.groups << " " << g.start << " " << g.count << "\\n"; }}\n'
    )
    binary = tmp_path / "check"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            f"-I{root / 'tools/qwen4exp'}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    rng = random.Random(44)
    values = [*ops.TEST_POSITIONS, *(rng.randrange(-(1 << 63), 1 << 63) for _ in range(1000))]
    run = subprocess.run([str(binary)], input="\n".join(map(str, values)), capture_output=True, text=True, check=True)
    actual = torch.tensor([[int(v) for v in line.split()] for line in run.stdout.splitlines()], dtype=torch.int64).T
    expected = ops.reference(torch.tensor(values, dtype=torch.int64), 4, (1 << 63) - 1)
    assert all(torch.equal(actual[i], expected[i]) for i in range(3))
    assert torch.all((actual[2] >= 0) & (actual[2] < 4))


@pytest.fixture
def resource(monkeypatch):
    monkeypatch.setattr(ops, "_on_npu", lambda device: device.type == "cpu")
    monkeypatch.setattr(ops, "_is_capturing", lambda: False)
    calls = []

    def launch(kernel, tensors, blocks):
        positions, packed, descriptor = tensors
        assert blocks == 1 and descriptor.tolist() == [len(positions)]
        packed.zero_()
        for row, value in enumerate(ops.reference(positions, 4, (1 << 63) - 1)):
            packed[row, : len(positions)].copy_(value)
        calls.append((kernel, tuple(tensors)))

    result = ops.QSAGeometry(object(), launch)
    result.prewarm(torch.device("cpu"))
    return result, calls


@pytest.mark.parametrize("rows", ops.ROWS)
@pytest.mark.parametrize("capacity", (-9, 0, 1, 512, 65536))
def test_wrapper_exact_changed_inputs_nonmutation_and_descriptor_stability(resource, rows, capacity):
    op, calls = resource
    config = op._configs[(rows, torch.device("cpu"))]
    for offset in range(0, len(ops.TEST_POSITIONS), rows):
        positions = torch.tensor(ops.TEST_POSITIONS[offset : offset + rows], dtype=torch.int64)
        original = positions.clone()
        output = op(positions, 4, capacity)
        assert all(torch.equal(a, b) for a, b in zip(output, ops.reference(positions, 4, capacity)))
        assert all(value.is_contiguous() for value in output)
        assert torch.equal(positions, original)
        assert calls[-1][1][-1] is config
    assert len(op._configs) == 2


def test_descriptor_missing_never_allocates_during_capture(resource, monkeypatch):
    op, _ = resource
    op._configs.clear()
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarm"):
        op(torch.zeros(3, dtype=torch.int64), 4, 512)
    assert op._configs == {}
    with pytest.raises(RuntimeError, match="eager"):
        op.prewarm(torch.device("cpu"))


def test_native_failure_retains_first_tensor_frames_and_rejects_retry(resource):
    op, _ = resource
    error = RuntimeError("submission uncertain")

    def fail(kernel, tensors, blocks):
        raise error

    op.launch = fail
    positions = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="uncertain"):
        op(positions, 4, 512)
    traceback = op.failure_traceback
    assert op.failed and op.failure is error and traceback is not None
    with pytest.raises(RuntimeError, match="failed"):
        op(positions, 4, 512)
    assert op.failure_traceback is traceback


def pinned_function():
    source = (Path(__file__).parent / "fixtures/qfn_pinned_qsa_geometry.txt").read_text()
    namespace = {"torch": torch}
    exec(compile(source, "<pinned-geometry>", "exec"), namespace)
    return namespace["_qsa_position_geometry"], source


def test_binding_canonical_target_exact_source_and_fallback(resource, monkeypatch):
    op, calls = resource
    original, source = pinned_function()
    assert hashlib.sha256(source.encode()).hexdigest() == binding.QUALIFIED_SOURCE_SHA256
    monkeypatch.setattr(binding.inspect, "getsource", lambda function: source)
    wrapped = binding.build_geometry(original, op)
    for positions, ratio in (
        (torch.zeros(3, dtype=torch.int64), 4),
        (torch.zeros(6, dtype=torch.int64), 4),
        (torch.zeros(1, dtype=torch.int64), 4),
        (torch.zeros(3, dtype=torch.int32), 4),
        (torch.zeros(3, dtype=torch.int64), 3),
        (torch.zeros(6, dtype=torch.int64)[::2], 4),
    ):
        output = wrapped(positions, ratio, 512)
        assert all(torch.equal(a, b) for a, b in zip(output, original(positions, ratio, 512)))
    assert len(calls) == 2
    op._configs.clear()
    wrapped(torch.zeros(3, dtype=torch.int64), 4, 512)
    assert len(calls) == 2
    assert len(binding.TARGETS) == 1 and binding.TARGETS[0].startswith("vllm_ascend.")
    monkeypatch.setattr(binding.inspect, "getsource", lambda function: source + "# changed\n")
    with pytest.raises(RuntimeError, match="exact pinned"):
        binding.build_geometry(original, op)


def complete_gate():
    cases = []
    for rows in ops.ROWS:
        for offset in range(0, len(ops.TEST_POSITIONS), rows):
            cases.append(dict(rows=rows, offset=offset, captured_exact=True, inputs_preserved=True))
            cases.extend(
                dict(rows=rows, offset=offset, capacity=capacity, eager_exact=True) for capacity in ops.TEST_CAPACITIES
            )
    return dict(failed=False, retired=True, receipt=dict(passed=True, rows=list(ops.ROWS), ratio=4, cases=cases))


def test_selection_reload_updates_existing_prefix_objects_and_canonical_binding(resource, monkeypatch):
    from tools.qwen4exp.resident_candidates import qsa_graph_selection as selection
    from vllm_ascend.models.qwen4_exp.ops import qsa_indexer

    original = qsa_indexer._qsa_position_geometry
    held_prefix = selection.prefix_selection
    importlib.reload(selection)
    assert held_prefix is not selection.prefix_selection
    assert held_prefix.__globals__ is selection.prefix_selection.__globals__
    calls = []

    def changed(positions, compress_ratio, capacity):
        calls.append((positions.clone(), compress_ratio, capacity))
        return original(positions, compress_ratio, capacity)

    monkeypatch.setattr(qsa_indexer, "_qsa_position_geometry", changed)
    positions = torch.tensor([-1, 2051, 262143], dtype=torch.int64)
    for function in (held_prefix, selection.prefix_selection):
        result = function(positions, compress_ratio=4, token_topk=2048, capacity_groups=512)
        assert torch.equal(result.group_counts, original(positions, 4, 512)[0])
    assert len(calls) == 2
    monkeypatch.setattr(qsa_indexer, "_qsa_position_geometry", original)
    payload = binding.make_replacements(resource[0], complete_gate())
    assert tuple(payload) == binding.TARGETS
    assert all(key.startswith("vllm_ascend.") for key in payload)
    monkeypatch.setattr(selection, "_qsa_position_geometry", original)
    with pytest.raises(RuntimeError, match="reload qsa_graph_selection"):
        binding.make_replacements(resource[0], complete_gate())


def test_private_receipt_requires_every_changed_input_and_capacity():
    gate = complete_gate()
    binding.require_private_gate(gate)
    gate["receipt"]["cases"].pop()
    with pytest.raises(RuntimeError, match="incomplete"):
        binding.require_private_gate(gate)
    gate = complete_gate()
    gate["retired"] = False
    with pytest.raises(RuntimeError, match="retired"):
        binding.require_private_gate(gate)


def test_gate_publishes_owner_before_capture_failure_and_keeps_frames(resource, monkeypatch):
    op, _ = resource
    sentinel = RuntimeError("stream allocation failed")

    def fail(**kwargs):
        raise sentinel

    monkeypatch.setattr(
        ops.torch, "npu", SimpleNamespace(current_stream=lambda device: object(), Stream=fail), raising=False
    )
    owner = {}
    with pytest.raises(RuntimeError, match="stream allocation"):
        ops.validate_graphs(op, torch.device("cpu"), owner=owner)
    assert owner["failed"] and owner["failure"] is sentinel and owner["failure_traceback"] is not None
    assert "retired" not in owner


@pytest.mark.parametrize("patch", ({"repeats": True}, {"replays": 0}, {"action": "reset"}, {"build": "relative"}))
def test_rpc_rejects_unsafe_requests(patch):
    request = dict(action="validate", build="/immutable", repeats=1, replays=1)
    request.update(patch)
    with pytest.raises(ValueError):
        binding.parse_request(json.dumps(request))


def test_rpc_preserves_unrelated_calls_and_reports_errors():
    original = lambda worker, output: (worker, output)
    wrapped = binding.make_rpc(original)
    worker = SimpleNamespace(rank=0)
    assert wrapped(worker, "regular") == (worker, "regular")
    assert wrapped(worker, '{"action":"reset"}')["passed"] is False


def test_verify_bundle_rejects_queue_drain_or_tampered_bytes(tmp_path):
    sources = {}
    for name in ops.REQUIRED_SOURCES:
        (tmp_path / name).write_bytes(name.encode())
        sources[name] = hashlib.sha256(name.encode()).hexdigest()
    (tmp_path / "bridge.so").write_bytes(b"bridge")
    (tmp_path / "native.bin").write_bytes(b"native")
    report = dict(
        namespace=ops.NAMESPACE,
        entry=ops.NAMESPACE,
        bridge_source="native_owned_bridge.cpp",
        flush_host_queue=False,
        eager_only=False,
        sources=sources,
        bridge=dict(path=str(tmp_path / "bridge.so"), sha256=hashlib.sha256(b"bridge").hexdigest()),
        binaries={
            "native_qsa_geometry_v2": dict(
                path=str(tmp_path / "native.bin"), sha256=hashlib.sha256(b"native").hexdigest()
            )
        },
    )
    manifest = tmp_path / "provenance.json"
    manifest.write_text(json.dumps(report))
    assert ops.verify_build(tmp_path) == report
    report["flush_host_queue"] = True
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="no-drain"):
        ops.verify_build(tmp_path)
    report["flush_host_queue"] = False
    manifest.write_text(json.dumps(report))
    (tmp_path / "native.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        ops.verify_build(tmp_path)


def test_rpc_load_failure_retains_partial_owner_and_rejects_retry(monkeypatch):
    monkeypatch.setattr(ops, "_is_capturing", lambda: False)
    monkeypatch.setattr(ops, "verify_build", lambda build: {})
    monkeypatch.setattr(
        binding,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(mem_get_info=lambda: (1 << 30, 2 << 30), current_device=lambda: 0),
            device=lambda *args: torch.device("cpu"),
        ),
    )
    error = RuntimeError("kernel construction failed")

    def load(build, *, owner):
        owner["partial"] = object()
        raise error

    monkeypatch.setattr(ops, "load_resources", load)
    worker = SimpleNamespace(model_runner=SimpleNamespace(execute_model_state=None))
    request = binding.parse_request('{"action":"preflight","build":"/immutable"}')
    assert binding.run_action(worker, request)["passed"]
    request["action"] = "load"
    with pytest.raises(RuntimeError, match="construction"):
        binding.run_action(worker, request)
    state = getattr(worker, binding.STATE_ATTRIBUTE)
    assert state["failed"] and "partial" in state["resources"] and state["failure"] is error
    first_traceback = state["failure_traceback"]
    with pytest.raises(RuntimeError, match="already owned or failed"):
        binding.run_action(worker, request)
    assert state["failure_traceback"] is first_traceback


def test_rpc_refuses_inflight_requests_before_allocation():
    worker = SimpleNamespace(model_runner=SimpleNamespace(execute_model_state=object()))
    with pytest.raises(RuntimeError, match="drain"):
        binding.run_action(worker, binding.parse_request('{"action":"load","build":"/immutable"}'))
    assert not hasattr(worker, binding.STATE_ATTRIBUTE)


def test_capture_audit_records_actual_dtype_without_replay_counters(resource, monkeypatch):
    op, _ = resource
    original, source = pinned_function()
    monkeypatch.setattr(binding.inspect, "getsource", lambda function: source)
    wrapped = binding.build_geometry(original, op)
    positions = torch.tensor([1, 2051, 262143], dtype=torch.int64)
    for _ in range(4):
        wrapped(positions, 4, 512)
    assert op.capture_contracts == []
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)
    for _ in range(4):
        wrapped(positions, 4, 512)
    wrapped(positions.to(torch.int32), 4, 512)
    assert len(op.capture_contracts) == 2
    assert op.capture_contracts[0] == dict(
        shape=[3], dtype="torch.int64", stride=[1], ratio=4, capacity=512, reason="ready"
    )
    assert op.capture_contracts[1]["reason"] == "dtype"
    for capacity in range(100):
        wrapped(positions, 4, capacity)
    assert len(op.capture_contracts) == binding.MAX_CAPTURE_CONTRACTS
    worker = SimpleNamespace()
    setattr(worker, binding.STATE_ATTRIBUTE, dict(loaded=True, failed=False, resources={"geometry": op}))
    receipt = binding.capture_inventory(worker)
    assert len(receipt["contracts"]) == binding.MAX_CAPTURE_CONTRACTS
    assert all("data_ptr" not in item and "count" not in item for item in receipt["contracts"])


def test_v2_source_guard_unwraps_installed_v1_without_changing_it(resource, monkeypatch):
    original, source = pinned_function()
    monkeypatch.setattr(binding.inspect, "getsource", lambda function: source if function is original else "bad")

    def v1(*args):
        return original(*args)

    v1._qfn_qsa_geometry_base = original
    wrapped = binding.build_geometry(v1, resource[0])
    assert wrapped._qfn_qsa_geometry_v2_base is original
    assert v1._qfn_qsa_geometry_base is original
    assert not hasattr(v1, "_qfn_qsa_geometry_v2_base")
