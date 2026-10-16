# SPDX-License-Identifier: Apache-2.0
"""Offline compiler simulation verifies source freezing and admission faults."""

import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as compiler
from tools.glm_perf import build_streaming as builder
from tools.glm_perf import streaming_memory
from tools.glm_perf.streaming_memory import SDK_CAPACITIES, memory_contract
from tools.glm_perf.streaming_protocol import canonical_digest, evidence_schema, file_digest


@pytest.fixture
def frozen_inputs(tmp_path):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    # Actual reference options, with small independently sealed fixture metadata.
    options = json.loads((Path(__file__).parent / "fixtures/streaming_reference_options.json").read_text())
    reference = {
        "version": 1021,
        "decode_version": 984,
        "build_options": options,
        "checkpoint": {"scale_layout": "fp16_storage_fp32_compute_v1"},
        "frozen_files": [],
    }
    reference["reference_sha256"] = canonical_digest(reference)
    reference_file = reference_dir / "reference.json"
    reference_file.write_text(json.dumps(reference))
    sdk_file = tmp_path / "sdk-header"
    sdk_file.write_text("offline SDK fixture; compilation is mocked")
    sdk = {
        "architecture": 2002,
        "capacities": dict(zip(("ub", "l1", "l0a", "l0b", "l0c"), SDK_CAPACITIES)),
        "files": [{"path": str(sdk_file), "sha256": file_digest(sdk_file)}],
    }
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(memory_contract(sdk)))
    return contract, reference_file


@pytest.fixture
def fake_compiler(monkeypatch, tmp_path):
    commands = []

    def compile_command(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1] if "-o" in command else command[2])
        output.write_bytes(b"offline compiler simulation\n" + output.name.encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(compiler.subprocess, "run", compile_command)
    monkeypatch.setattr(compiler, "include_paths", lambda: [])
    monkeypatch.setattr(compiler, "library_paths", lambda: [])
    monkeypatch.setattr(
        compiler.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "init.py"))
    )
    return commands


@pytest.fixture
def candidate(tmp_path, frozen_inputs, fake_compiler):
    contract, reference = frozen_inputs
    result = builder.build(
        contract, reference, tmp_path / "candidate", version=1102, cann_root=tmp_path, source_root=tmp_path
    )
    return result


def test_both_projections_generic_and_w4_compile_streaming_and_freeze_all_headers(candidate, fake_compiler):
    projected = [
        command for command in fake_compiler if len(command) > 2 and Path(command[1]).name == "glm_fused_moe.cpp"
    ]
    assert len(projected) == 4
    for command in projected:
        assert "-DGLM_STREAMING_PIPELINE" in command
        assert Path(command[1]).parent == candidate / "native_sources"
        assert f"-I{candidate / 'native_sources'}" in command
    provenance = json.loads((candidate / "provenance.json").read_text())
    for header in builder.STREAMING_HEADERS:
        assert file_digest(candidate / "native_sources" / header) == provenance[header]["source_sha256"]
    identity = builder.admit_candidate(candidate)
    assert identity == builder.candidate_identity(candidate)
    assert (
        identity["reference_sha256"]
        == json.loads((candidate / "reference/reference.json").read_text())["reference_sha256"]
    )
    assert identity["contract_sha256"] == file_digest(candidate / "contract.json")


@pytest.mark.parametrize("flag", [1, "true", None])
def test_streaming_flag_requires_boolean_before_any_artifact(tmp_path, flag):
    with pytest.raises(ValueError, match="boolean"):
        compiler.build(tmp_path / "bad", tmp_path, tmp_path, streaming_pipeline=flag)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "option", ["fp16_weight_scales", "direct_w4_l1", "pair_prefill_scale_groups", "product_pipe_events"]
)
def test_missing_required_layout_rejected_before_artifact(tmp_path, frozen_inputs, option):
    options = json.loads(frozen_inputs[1].read_text())["build_options"]
    accepted = inspect.signature(compiler.build).parameters
    kwargs = {key: value for key, value in options.items() if key in accepted and key != "version"}
    kwargs[option] = False
    with pytest.raises(ValueError):
        compiler.build(tmp_path / "bad", tmp_path, tmp_path, streaming_pipeline=True, **kwargs)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "path",
    [
        "native_sources/glm_streaming_products.h",
        "glm_fused_down_w4.bin",
        "glm_reconstruction_v1102_helpers/glm_fused_moe.py",
        "contract.json",
    ],
)
def test_changed_sources_binaries_runtime_helpers_and_proof_rejected(candidate, path):
    file = candidate / path
    file.write_bytes(file.read_bytes() + b"\nchanged")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        builder.admit_candidate(candidate)


def test_pointer_consumer_source_is_frozen_and_semantic_tamper_rejects_admission(candidate):
    source_name = "native_sources/glm_streaming_products_pointer.h"
    pointer = candidate / source_name
    provenance = json.loads((candidate / "provenance.json").read_text())
    assert provenance["_streaming_sources"][source_name] == file_digest(pointer)
    caller = (candidate / "native_sources/glm_fused_moe.cpp").read_text()
    assert '#include "glm_streaming_products_pointer.h"' in caller
    assert "GlmStreamingProductsPointer::ConsumeDirect(" in caller
    assert builder.admit_candidate(candidate)["version"] == 1102
    original = pointer.read_text()
    assert "ResetMask();" in original
    # This changes the caller-boundary mask contract without touching binaries,
    # proving the frozen source map covers the actual newly integrated header.
    pointer.write_text(original.replace("ResetMask();", "// full-mask restoration removed", 1))
    with pytest.raises(ValueError, match="frozen sources or binaries changed"):
        builder.admit_candidate(candidate)


def test_missing_projection_binary_is_not_complete(candidate):
    (candidate / "glm_fused_down.bin").unlink()
    with pytest.raises(ValueError, match="binaries"):
        builder.admit_candidate(candidate)


def test_build_evidence_missing_coverage_rejected(candidate):
    path = candidate / "build.json"
    evidence = json.loads(path.read_text())
    evidence["coverage"].remove("product_consumer")
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="coverage"):
        builder.admit_candidate(candidate)


def test_source_freezing_does_not_depend_on_modified_workspace(candidate, monkeypatch):
    monkeypatch.setattr(compiler, "HERE", Path("/missing/workspace"))
    assert builder.admit_candidate(candidate)["version"] == 1102


def test_append_only_paths_and_version_reservations(candidate, frozen_inputs, tmp_path):
    contract, reference = frozen_inputs
    with pytest.raises(FileExistsError, match="append-only"):
        builder.build(contract, reference, candidate, version=1103, cann_root=tmp_path, source_root=tmp_path)
    with pytest.raises(ValueError, match="namespace"):
        builder.build(contract, reference, tmp_path / "other", version=1102, cann_root=tmp_path, source_root=tmp_path)
    assert not (tmp_path / "other").exists()


def test_compile_failure_never_writes_complete_evidence_and_does_not_recycle_version(
    tmp_path, frozen_inputs, monkeypatch
):
    contract, reference = frozen_inputs

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "offline compiler fault")

    monkeypatch.setattr(compiler.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        builder.build(contract, reference, tmp_path / "failed", version=1102, cann_root=tmp_path, source_root=tmp_path)
    assert not (tmp_path / "failed/build.json").exists()
    assert json.loads((tmp_path / "build-failure-v1102.json").read_text())["state"] == "failed"
    with pytest.raises(FileExistsError):
        builder.build(contract, reference, tmp_path / "retry", version=1102, cann_root=tmp_path, source_root=tmp_path)


def test_invalid_memory_proof_does_not_compile(tmp_path, frozen_inputs, fake_compiler):
    contract, reference = frozen_inputs
    proof = json.loads(contract.read_text())
    proof["projections"]["down_generic"]["ub_total"] += 16
    contract.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="memory contract"):
        builder.build(contract, reference, tmp_path / "bad", version=1102, cann_root=tmp_path, source_root=tmp_path)
    assert not fake_compiler


def native_report(candidate):
    identity = builder.candidate_identity(candidate)
    data = candidate.parent / "native-gate-data.json"
    data.write_text("offline gate fixture; no hardware execution")
    report = {
        "schema_version": 1,
        "kind": "native_gates",
        "state": "complete",
        "identity": identity,
        "dispatch": dict(builder.DISPATCH),
        "coverage": evidence_schema()["x-required-coverage"]["native_gates"],
        "artifacts": [{"path": data.name, "sha256": file_digest(data)}],
        "errors": [],
        "result": {key: True for key in evidence_schema()["x-complete-result-rules"]["native_gates"]},
    }
    report["result"]["resident_memory_admission"] = {
        "passed": True,
        "required_free_bytes": 8192,
        "includes": sorted(builder.MEMORY_COMPONENTS),
    }
    return report


def test_probe_allowed_but_live_requires_native_gate(candidate):
    assert builder.admit_candidate(candidate)
    with pytest.raises(ValueError, match="native gates"):
        builder.admit_candidate(candidate, live=True)


@pytest.mark.parametrize(
    "fault", ["none", "copied_identity", "missing_case", "changed_artifact", "missing_memory", "incomplete_memory"]
)
def test_live_admission_checks_native_identity_coverage_artifacts_and_memory(candidate, fault):
    report = native_report(candidate)
    if fault == "copied_identity":
        report["identity"]["source_sha256"] = "a" * 64
    elif fault == "missing_case":
        report["coverage"].pop()
    elif fault == "changed_artifact":
        (candidate.parent / "native-gate-data.json").write_text("changed")
    elif fault == "missing_memory":
        report["result"].pop("resident_memory_admission")
    elif fault == "incomplete_memory":
        report["result"]["resident_memory_admission"]["includes"].remove("verification")
    path = candidate.parent / "native-gates.json"
    path.write_text(json.dumps(report))
    if fault == "none":
        assert builder.admit_candidate(candidate, native_evidence_path=path, live=True)
    else:
        with pytest.raises(ValueError):
            builder.admit_candidate(candidate, native_evidence_path=path, live=True)


@pytest.fixture
def changed_workspace_header(tmp_path, monkeypatch):
    workspace = tmp_path / "different-workspace"
    workspace.mkdir()
    header = workspace / "glm_streaming_contract.h"
    header.write_text(
        Path(streaming_memory.__file__).with_name("glm_streaming_contract.h").read_text()
        + "\n// different workspace header, not the frozen candidate header\n"
    )
    # Substitute the default lookup root without touching shared source files.
    monkeypatch.setattr(streaming_memory, "__file__", str(workspace / "streaming_memory.py"))
    return header


def test_intact_frozen_candidate_ignores_changed_current_workspace_header(candidate, changed_workspace_header):
    proof = json.loads((candidate / "contract.json").read_text())
    assert memory_contract()["header_sha256"] == file_digest(changed_workspace_header)
    assert proof["header_sha256"] != file_digest(changed_workspace_header)
    assert builder.admit_candidate(candidate)["version"] == 1102


@pytest.mark.parametrize("fault", ["frozen_header", "proof_hash", "proof_bounds"])
def test_frozen_header_and_proof_still_verified_with_changed_workspace(candidate, changed_workspace_header, fault):
    if fault == "frozen_header":
        path = candidate / "native_sources/glm_streaming_contract.h"
        path.write_bytes(path.read_bytes() + b"\nchanged frozen header")
    else:
        path = candidate / "contract.json"
        proof = json.loads(path.read_text())
        if fault == "proof_hash":
            proof["header_sha256"] = file_digest(changed_workspace_header)
        else:
            proof["projections"]["down_generic"]["ub_total"] += 16
        path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="memory contract"):
        builder.admit_candidate(candidate)


def test_current_memory_api_bounds_remain_authoritative_for_frozen_admission(candidate, monkeypatch):
    monkeypatch.setattr(streaming_memory, "SDK_RESERVE", streaming_memory.SDK_RESERVE + 32)
    with pytest.raises(ValueError, match="memory contract"):
        builder.admit_candidate(candidate)


def test_new_build_still_binds_current_workspace_header(
    tmp_path, frozen_inputs, fake_compiler, changed_workspace_header
):
    contract, reference = frozen_inputs
    with pytest.raises(ValueError, match="memory contract"):
        builder.build(
            contract, reference, tmp_path / "new-build", version=1103, cann_root=tmp_path, source_root=tmp_path
        )
    assert not fake_compiler
    assert not (tmp_path / "new-build").exists()
