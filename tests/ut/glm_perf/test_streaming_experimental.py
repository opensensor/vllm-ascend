# SPDX-License-Identifier: Apache-2.0
"""Pending diagnostic loading remains distinct from qualified live admission."""

import copy
import json
from pathlib import Path

import pytest

from tools.glm_perf import fused_moe_control as control
from tools.glm_perf.streaming_protocol import file_digest, native_case_ids


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    namespace = "glm_reconstruction_v1102"
    package = namespace + "_helpers"
    helpers = build / package
    helpers.mkdir()
    (helpers / "__init__.py").write_text("# offline fixture\n")
    names = (
        "glm_reconstruction_bridge_v1102.so",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_pack.bin",
    )
    for name in names:
        (build / name).write_bytes(b"offline diagnostic fixture: " + name.encode())
    identity = {
        "contract_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "binaries": {name: file_digest(build / name) for name in names},
        "version": 1102,
        "namespace": namespace,
    }
    options = {
        "namespace": namespace,
        "version": 1102,
        "helper_package": package,
        "streaming_pipeline": True,
        "prepared_weight_layout": True,
    }
    provenance = {"_build": options, "_helpers": {"__init__.py": file_digest(helpers / "__init__.py")}}
    (build / "provenance.json").write_text(json.dumps(provenance))
    records = [
        {
            "id": f"w{bits}_a4_k2048_rows5_balanced",
            "passed": True,
            "same_input_bitwise": True,
            "finite": True,
            "padding_poison_independent": True,
            "all_peer_zero": True,
            "changed_replays": 4,
        }
        for bits in (2, 3, 4)
    ]
    assert {row["id"] for row in records}.issubset(native_case_ids())
    dispatch = {
        "bulk_a4_gate": "streaming",
        "bulk_a4_down": "streaming",
        "sparse": "qualified_reference",
        "a8": "qualified_reference",
        "decode_version": 984,
    }
    report = {
        "schema_version": 1,
        "kind": "native_gates",
        "state": "pending",
        "identity": copy.deepcopy(identity),
        "dispatch": dispatch,
        "coverage": [row["id"] for row in records],
        "artifacts": [],
        "errors": [],
        "result": {"hardware_validated": False, "experimental_only": True},
    }
    records_file = tmp_path / "diagnostic-records.json"
    report_file = tmp_path / "diagnostic.json"

    def write(*, include_records=True):
        records_file.write_text(json.dumps(records))
        artifact = records_file if include_records else tmp_path / "not-comparison.json"
        if not include_records:
            artifact.write_text("{}")
        report["artifacts"] = [{"path": artifact.name, "sha256": file_digest(artifact)}]
        report_file.write_text(json.dumps(report))

    admitted = []

    def probe_admission(root, **kwargs):
        assert Path(root) == build
        admitted.append(kwargs)
        return copy.deepcopy(identity)

    monkeypatch.setattr(control, "admit_candidate", probe_admission)
    write()
    return build, report_file, records_file, report, records, write, admitted


def test_pending_success_builds_miniqualification_without_claiming_complete_gates(experiment):
    build, report_file, _, report, _, _, calls = experiment
    result = control.experimental_manifest(build, report_file)
    assert calls == [{"live": False}]
    assert report["state"] == "pending"
    assert report["result"].get("bitwise_equal") is None
    payload = result.value
    compile(payload["validation_source"], "experimental validation", "exec")
    assert payload["operators"] == ["glm_reconstruction_v1102::launch"]
    assert "for activation in (8,4) for bits in (2,3,4)" in payload["validation_source"]
    assert "gate_case" in payload["validation_source"]
    assert "'model_quality':'not_evaluated'" in payload["validation_source"]


@pytest.mark.parametrize("state", ["complete", "failed", "blocked"])
def test_experiment_requires_pending_evidence(experiment, state):
    build, path, _, report, _, write, _ = experiment
    report["state"] = state
    if state in ("failed", "blocked"):
        report["errors"] = ["offline diagnostic failure"]
    write()
    with pytest.raises(ValueError):
        control.experimental_manifest(build, path)


@pytest.mark.parametrize(
    "field", ["passed", "same_input_bitwise", "finite", "padding_poison_independent", "all_peer_zero"]
)
@pytest.mark.parametrize("value", [False, 1, None])
def test_every_diagnostic_case_requires_strict_success_flags(experiment, field, value):
    build, path, _, _, records, write, _ = experiment
    records[1][field] = value
    write()
    with pytest.raises(ValueError, match="parity/replay"):
        control.experimental_manifest(build, path)


@pytest.mark.parametrize("replays", [0, 1, 2])
def test_insufficient_changed_replays_are_rejected(experiment, replays):
    build, path, _, _, records, write, _ = experiment
    records[0]["changed_replays"] = replays
    write()
    with pytest.raises(ValueError, match="parity/replay"):
        control.experimental_manifest(build, path)


def test_missing_replay_result_is_rejected(experiment):
    build, path, _, _, records, write, _ = experiment
    records[0].pop("changed_replays")
    write()
    with pytest.raises(ValueError, match="parity/replay"):
        control.experimental_manifest(build, path)


@pytest.mark.parametrize(
    "fault", ["empty_records", "missing_record", "different_record_id", "missing_artifact", "empty_coverage"]
)
def test_coverage_requires_actual_matching_comparison_records(experiment, fault):
    build, path, _, report, records, write, _ = experiment
    if fault == "empty_records":
        records.clear()
    elif fault == "missing_record":
        records.pop()
    elif fault == "different_record_id":
        records[0]["id"] = "w2_a4_k4096_rows5_balanced"
    elif fault == "empty_coverage":
        report["coverage"] = []
    write(include_records=fault != "missing_artifact")
    with pytest.raises(ValueError):
        control.experimental_manifest(build, path)


@pytest.mark.parametrize("field", ["contract_sha256", "reference_sha256", "source_sha256", "binaries", "version"])
def test_copied_or_stale_diagnostic_identity_is_rejected(experiment, field):
    build, path, _, report, _, write, _ = experiment
    if field == "binaries":
        report["identity"][field]["glm_fused_down.bin"] = "e" * 64
    elif field == "version":
        report["identity"][field] = 1103
        report["identity"]["namespace"] = "glm_reconstruction_v1103"
    else:
        report["identity"][field] = "e" * 64
    write()
    with pytest.raises(ValueError):
        control.experimental_manifest(build, path)


def test_changed_diagnostic_artifact_rejected_before_resource_generation(experiment):
    build, path, records_file, _, _, _, _ = experiment
    records_file.write_text("[]")
    with pytest.raises(ValueError, match="artifact"):
        control.experimental_manifest(build, path)


@pytest.mark.parametrize("name", ["glm_fused_gate_up.bin", "glm_fused_down.bin", "glm_reconstruction_bridge_v1102.so"])
def test_changed_binary_is_rejected_even_with_mocked_build_admission(experiment, name):
    build, path, _, _, _, _, _ = experiment
    (build / name).write_bytes(b"changed offline binary")
    with pytest.raises(ValueError, match="exact binaries"):
        control.experimental_manifest(build, path)


def test_changed_frozen_runtime_helper_is_rejected(experiment):
    build, path, _, _, _, _, _ = experiment
    (build / "glm_reconstruction_v1102_helpers/__init__.py").write_text("changed")
    with pytest.raises(ValueError, match="helpers differ"):
        control.experimental_manifest(build, path)


def test_pending_diagnostic_cannot_enter_ordinary_qualified_manifest(experiment, monkeypatch):
    build, path, _, _, _, _, _ = experiment

    def qualified_admission(*args, **kwargs):
        assert kwargs["live"] is True
        raise ValueError("complete native gates required")

    monkeypatch.setattr(control, "admit_candidate", qualified_admission)
    with pytest.raises(ValueError, match="complete native gates required"):
        control.manifest(build, path, streaming_gate_report=path)


def test_matching_but_unknown_diagnostic_case_ids_are_rejected(experiment):
    build, path, _, report, records, write, _ = experiment
    records[0]["id"] = "unrecognized-diagnostic-shape"
    report["coverage"][0] = records[0]["id"]
    write()
    with pytest.raises(ValueError):
        control.experimental_manifest(build, path)
