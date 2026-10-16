# SPDX-License-Identifier: Apache-2.0
"""CPU bounds/coverage tests; device gate execution remains coordinator-owned."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.glm_perf import test_streaming_build as shared_build
from tools.glm_perf import build_streaming
from tools.glm_perf import streaming_probe as probe
from tools.glm_perf.streaming_protocol import evidence_schema, file_digest


def measured(free=3559636992):
    return {
        "free_bytes": free,
        "resident_allocated_bytes": 40123947520,
        "runtime_context_bytes": 1610612736,
        "safety_bytes": 268435456,
        "raw_provenance": [{"path": "context-measurement.json", "sha256": "a" * 64}],
    }


def test_envelope_includes_whole_second_process_and_both_full_geometry_scratch():
    result = probe.memory_envelope(measured())
    assert result["passed"]
    assert result["required_free_bytes"] == 3431612420
    assert result["resident_already_allocated_bytes"] == measured()["resident_allocated_bytes"]
    assert result["components"]["reference_scratch"] == result["components"]["candidate_scratch"] == 178921474
    assert result["components"]["bank_fixtures"] == 905969664
    assert result["resident_memory_admission"]["passed"]
    assert set(result["resident_memory_admission"]["includes"]) == probe.MEMORY_COMPONENTS
    assert not probe.memory_envelope(measured(result["required_free_bytes"] - 1))["passed"]
    assert probe.memory_envelope(measured(result["required_free_bytes"]))["passed"]


@pytest.mark.parametrize("name", ["free_bytes", "resident_allocated_bytes", "runtime_context_bytes", "safety_bytes"])
@pytest.mark.parametrize("value", [None, False, 0, -1, 1.25])
def test_unmeasured_or_invalid_memory_is_rejected(name, value):
    evidence = measured()
    evidence[name] = value
    with pytest.raises(ValueError, match=name):
        probe.memory_envelope(evidence)


@pytest.mark.parametrize("raw", [None, {}, []])
def test_raw_measurement_provenance_is_required(raw):
    evidence = measured()
    evidence["raw_provenance"] = raw
    with pytest.raises(ValueError, match="provenance"):
        probe.memory_envelope(evidence)


def test_complete_required_cases_and_dense_smoke_without_coverage_reduction():
    rows = probe.cases()
    assert len(rows) == 438
    assert {row["id"] for row in rows} == set(evidence_schema()["x-required-coverage"]["native_gates"])
    assert rows[0]["id"] == "w4_a4_k4096_rows5_balanced"
    assert rows[0]["tokens"] == 45
    assert all(17 <= row["tokens"] <= 1280 for row in rows)


def test_selected_diagnostic_cases_preserve_requested_ids_and_required_coverage():
    names = ["w4_a4_k4096_rows31_balanced", "w3_a4_k4096_rows31_balanced", "w2_a4_k4096_rows31_balanced"]
    assert [case["id"] for case in probe.select_cases(names)] == names
    assert len(probe.select_cases()) == 438
    assert probe.select_cases(limit=1)[0]["id"] == "w4_a4_k4096_rows5_balanced"
    with pytest.raises(ValueError, match="mutually exclusive"):
        probe.select_cases(names, limit=1)
    with pytest.raises(ValueError, match="unique frozen"):
        probe.select_cases(["unknown"])
    with pytest.raises(ValueError, match="unique frozen"):
        probe.select_cases([names[0], names[0]])


@pytest.mark.parametrize("case", probe.cases(), ids=lambda row: row["id"])
def test_routes_match_initial_expert_density_and_replay_boundaries(case):
    ids = sum(probe.route_ids(case), [])
    assert len(ids) == case["tokens"] * 8
    counts = Counter(value for value in ids if 0 <= value < 72)
    if case["pattern"] == "all_peer" or case["rows"] == 0:
        assert not counts
    elif case["rows"] is None:
        assert max(counts.values()) - min(counts.values()) <= 1
    elif case["pattern"] == "balanced":
        assert len(counts) == 72
        assert set(counts.values()) == {case["rows"]}
    else:
        assert counts == {0: case["rows"]}
    changed = sum(probe.route_ids(case, replay=1), [])
    assert len(changed) == len(ids)
    assert all(0 <= value <= 73 for value in changed)
    assert changed != ids


def test_output_gate_requires_exact_fp32_bits_and_finite_values():
    probe._same(torch, torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    with pytest.raises(AssertionError, match="bits differ"):
        probe._same(torch, torch.tensor([0.0]), torch.tensor([-0.0]))
    with pytest.raises(AssertionError, match="nonfinite"):
        probe._same(torch, torch.tensor([float("nan")]), torch.tensor([float("nan")]))


def test_insufficient_memory_stops_before_worker_or_subprocess(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"reference_sha256": "b" * 64, "checkpoint": {"descriptors": {}}}))
    raw = tmp_path / "context-measurement.json"
    raw.write_text("measured raw context\n")
    evidence = measured(100)
    evidence["raw_provenance"][0]["sha256"] = file_digest(raw)
    memory = tmp_path / "memory-evidence.json"
    memory.write_text(json.dumps(evidence))
    output = tmp_path / "gates.json"
    monkeypatch.setattr(probe, "admit_candidate", lambda *a, **kw: {"reference_sha256": "b" * 64})
    monkeypatch.setattr(probe, "_verify_reference_build", lambda *a: None)
    forbidden = lambda *a, **kw: pytest.fail("blocked memory proof reached execution")
    monkeypatch.setattr(probe, "worker", forbidden)
    monkeypatch.setattr(probe.subprocess, "run", forbidden)
    argv = [
        "probe",
        "--build-dir",
        str(tmp_path),
        "--reference",
        str(reference),
        "--reference-build-dir",
        str(tmp_path),
        "--checkpoint",
        str(tmp_path),
        "--memory-evidence",
        str(memory),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    probe.main()
    report = json.loads(output.read_text())
    assert report["state"] == "blocked"
    assert report["coverage"] == []
    assert "no device initialization" in report["errors"][0]


def test_eager_stage_timing_restores_launch_before_graph_capture():
    calls = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            calls.append("record")

        def elapsed_time(self, other):
            return 1.5

    class Native:
        gate_kernel = object()

        def __init__(self):
            self.launch = lambda *args: calls.append("launch")

        def __call__(self, *args):
            self.launch(self.gate_kernel, [], 8)
            return "output"

    native = Native()
    original = native.launch
    fake_torch = SimpleNamespace(npu=SimpleNamespace(Event=Event, synchronize=lambda: calls.append("sync")))
    output, metrics = probe._eager_stages(native, (), fake_torch)
    assert output == "output"
    assert calls == ["record", "launch", "record", "sync"]
    assert metrics == [{"stage": "gate_kernel", "device_ms": 1.5}]
    assert native.launch is original


def test_published_paths_survive_nested_gate_directory_and_reader_cwd(tmp_path, monkeypatch):
    """Exercise real reader/schema/identity validation with CPU compiler output."""
    from tools.glm_perf.streaming_resident import load_candidate

    contract, reference_path = shared_build.frozen_inputs.__wrapped__(tmp_path)
    shared_build.fake_compiler.__wrapped__(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    relative_build = Path("artifacts/T6/build-v1111")
    root = build_streaming.build(
        contract, reference_path, relative_build, version=1111, cann_root=tmp_path, source_root=tmp_path
    )
    identity = build_streaming.admit_candidate(root)
    nested_output = Path("artifacts/T10/qualified-v1111/gates.json")
    nested_output.parent.mkdir(parents=True)
    legacy = nested_output.with_name("gates-legacy.json")
    legacy.write_text(json.dumps({"complete": True, "test_fixture": "CPU compiler/path validation"}))
    proof = nested_output.with_name("native-proof.json")
    proof.write_text("CPU path regression fixture\n")
    formal_artifacts = [{"path": path.name, "sha256": file_digest(path)} for path in (proof, legacy)]
    memory = {"passed": True, "required_free_bytes": 42, "includes": sorted(probe.MEMORY_COMPONENTS)}
    report = {
        "schema_version": 1,
        "kind": "native_gates",
        "state": "complete",
        "identity": identity,
        "dispatch": dict(probe.DISPATCH),
        "coverage": evidence_schema()["x-required-coverage"]["native_gates"],
        "artifacts": formal_artifacts,
        "errors": [],
        "result": {
            **{key: True for key in evidence_schema()["x-complete-result-rules"]["native_gates"]},
            "resident_memory_admission": memory,
            **probe.report_artifact_paths(relative_build, legacy),
        },
    }
    for key in ("build_dir", "build_evidence", "legacy_gate_report"):
        assert Path(report["result"][key]).is_absolute()
    assert report["artifacts"] == formal_artifacts
    calls = []

    def manifest_factory(build_path, legacy_path, *, streaming_gate_report):
        assert build_path == root.resolve()
        assert legacy_path == legacy.resolve()
        assert streaming_gate_report == nested_output
        assert json.loads(legacy_path.read_text())["complete"] is True
        calls.append("manifest")
        return SimpleNamespace(verify_files=lambda: calls.append("verify"))

    reference = json.loads(reference_path.read_text())
    nested_output.write_text(json.dumps(report))
    candidate = load_candidate(reference, nested_output, manifest_factory=manifest_factory)
    assert candidate.build_dir == root.resolve()
    assert calls == ["manifest", "verify"]
    # Real admission is left active: this also verifies every frozen source,
    # helper and binary digest, complete coverage, and all artifact bytes.
    assert build_streaming.admit_candidate(root, native_evidence_path=nested_output, live=True) == identity

    # Reproduce the original producer bug without a reader CWD fallback.
    broken = json.loads(json.dumps(report))
    broken["result"].update(
        build_dir=str(relative_build), build_evidence=str(relative_build / "build.json"), legacy_gate_report=str(legacy)
    )
    nested_output.write_text(json.dumps(broken))
    with pytest.raises(FileNotFoundError):
        load_candidate(reference, nested_output, manifest_factory=manifest_factory)
    assert calls == ["manifest", "verify"]

    # The fixed report also works when the caller no longer uses producer CWD.
    nested_output.write_text(json.dumps(report))
    absolute_output = nested_output.resolve()
    absolute_legacy = legacy.resolve()
    absolute_root = root.resolve()
    elsewhere = tmp_path / "reader-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    def relocated_manifest(build_path, legacy_path, *, streaming_gate_report):
        assert build_path == absolute_root
        assert legacy_path == absolute_legacy
        assert streaming_gate_report == absolute_output
        return SimpleNamespace(verify_files=lambda: None)

    assert load_candidate(reference, absolute_output, manifest_factory=relocated_manifest).build_dir == absolute_root


def test_publication_rejects_missing_evidence_artifacts(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}")
    with pytest.raises(FileNotFoundError):
        probe.report_artifact_paths(build, legacy)
