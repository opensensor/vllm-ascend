# SPDX-License-Identifier: Apache-2.0
"""Reproduction snapshots preserve isolation and do not overwrite evidence."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools.sidecard import layer_runtime
from tools.sidecard.execution_options import BuildReference
from tools.sidecard.layer_runtime import prepare, stop


def test_launcher_keeps_venv_interpreter_symlink(tmp_path, monkeypatch):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv", ["launcher", "--output", str(output), "--specification", str(spec), "--cuda-python", str(python)]
    )
    calls = []
    monkeypatch.setattr(layer_runtime.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        layer_runtime.os, "execve", lambda executable, args, environment: calls.append((executable, args, environment))
    )
    layer_runtime.main()
    assert calls[0][0] == str(python) and calls[0][1][0] == str(python)
    assert calls[0][0] != str(python.resolve())
    assert json.loads((output / "process.json").read_text())["command"][0] == str(python)


def test_snapshot_excludes_environment_and_dirty_glm_experiments(tmp_path):
    source = Path(__file__).resolve().parents[3]
    output = tmp_path / "run"
    specification = {"npu_environment": "/private/runtime-environment.json"}
    configuration = prepare(output, specification, source)
    assert json.loads(configuration.read_text())["npu_environment"] == specification["npu_environment"]
    copied = list((output / "source" / "tools" / "glm_perf").glob("*"))
    assert sorted(path.name for path in copied) == [
        "__init__.py",
        "deployment_profiles.py",
        "fused_moe_profile.py",
        "glm_int4.py",
    ]
    assert not (output / "runtime-environment.json").exists()
    assert "tools/sidecard/layer_server.py" in json.loads((output / "source-sha256.json").read_text())
    assert "tools/sidecard/pcie/peer.py" in json.loads((output / "source-sha256.json").read_text())
    assert "tools/sidecard/cuda_oracle.cu" in json.loads((output / "source-sha256.json").read_text())
    assert (output / "source/tools/sidecard/cuda_oracle.cu").read_bytes() == (
        source / "tools/sidecard/cuda_oracle.cu"
    ).read_bytes()
    assert (output / "source/tools/sidecard/pcie/__init__.py").is_file()
    public_fixture = layer_runtime.ACCEPTANCE_FIXTURE
    assert (output / "source" / public_fixture).read_bytes() == (source / public_fixture).read_bytes()
    assert str(public_fixture) in json.loads((output / "source-sha256.json").read_text())
    with pytest.raises(FileExistsError):
        prepare(output, specification, source)


def test_snapshot_rejects_output_inside_source(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prepare(tmp_path / "artifacts", {}, tmp_path)


@pytest.mark.parametrize("transport", ["stage_transport", "expert_transport"])
def test_snapshot_rejects_overlong_unix_path_before_creating_output(tmp_path, transport):
    source = Path(__file__).resolve().parents[3]
    output = tmp_path / ("x" * layer_runtime.UNIX_SOCKET_PATH_BYTES)
    with pytest.raises(ValueError, match="too long"):
        prepare(output, {transport: "direct-dma"}, source)
    assert not output.exists()


def test_snapshot_missing_acceptance_fixture_fails_without_partial_output(tmp_path):
    source = tmp_path / "source"
    (source / "tools/sidecard").mkdir(parents=True)
    (source / "tools/sidecard/execution_acceptance.py").write_text("# fixture required\n")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="public acceptance fixture"):
        prepare(output, {}, source)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["native", "multi-route"])
def test_runtime_snapshot_retains_authenticated_native_source_bytes(tmp_path, monkeypatch, kind):
    source = Path(__file__).resolve().parents[3]
    build = tmp_path / "build"
    build.mkdir()

    def write(relative, content):
        path = build / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    native_source = b"// Frozen candidate kernel; live research is excluded.\n"
    if kind == "native":
        receipt_name, option, prefix = "execution-build.json", "native_build", "execution-source"
        manifest = {
            "source_files": {"csrc/candidate.cpp": write("source/csrc/candidate.cpp", native_source)},
            "source_manifest_sha256": write("source-manifest.json", b"source inventory"),
            "abi_manifest_sha256": write("abi-manifest.json", b"ABI inventory"),
        }
        manifest["source_files"]["tools/protoc"] = write("source/tools/protoc", b"executable fixture")
        (build / "source/tools/protoc").chmod(0o750)
    else:
        receipt_name, option, prefix = "multi-route-build.json", "multi_route", "multi-route-source"
        manifest = {
            "source_sha256": {"csrc/candidate.cpp": write("csrc/candidate.cpp", native_source)},
            "patch_sha256": write("candidate.patch", b"patch"),
            "provenance_sha256": write("provenance.json", b"provenance"),
        }
    checksum = write(receipt_name, json.dumps(manifest).encode())
    options = layer_runtime.validate_options({})
    monkeypatch.setattr(
        layer_runtime, "validate_options", lambda spec: replace(options, **{option: BuildReference(build, checksum)})
    )
    output = tmp_path / "frozen"
    prepare(output, {}, source)
    relative = f"native-inputs/{prefix}/csrc/candidate.cpp"
    assert (output / "source" / relative).read_bytes() == native_source
    if kind == "native":
        executable = output / "source/native-inputs/execution-source/tools/protoc"
        assert executable.stat().st_mode & 0o777 == 0o750
    assert (
        json.loads((output / "source-sha256.json").read_bytes())[relative] == hashlib.sha256(native_source).hexdigest()
    )
    (build / ("source/csrc/candidate.cpp" if kind == "native" else "csrc/candidate.cpp")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        list(layer_runtime.native_inputs(BuildReference(build, checksum), receipt_name))


def test_stop_rejects_a_pid_that_has_been_reused(tmp_path):
    (tmp_path / "process.json").write_text(json.dumps({"pid": os.getpid()}))
    with pytest.raises(ValueError, match="no longer belongs"):
        stop(tmp_path)


@pytest.mark.parametrize("parent_offset,return_code", [(0, 0), (1, 1)])
def test_actor_parent_guard_fails_closed_after_parent_exit(parent_offset, return_code):
    code = (
        f"import os; from tools.sidecard.layer_actor import guard_parent; guard_parent(os.getppid() + {parent_offset})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == return_code
    if return_code:
        assert "coordinator exited" in result.stderr


def test_snapshot_freezes_authenticated_expert_ownership(tmp_path):
    source = Path(__file__).resolve().parents[3]
    ownership = tmp_path / "ownership.json"
    ownership.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "checkpoint",
                "gpu_hot_experts": {str(i): [1, 71, 146, 280] for i in range(18, 33)},
            }
        )
    )
    original = ownership.read_bytes()
    specification = {
        "architecture": "cuda-backbone-ascend-experts",
        "checkpoint_sha256": "checkpoint",
        "expert_placement": str(ownership),
        "expert_placement_sha256": hashlib.sha256(original).hexdigest(),
    }
    config = prepare(tmp_path / "frozen", specification, source)
    frozen = Path(json.loads(config.read_bytes())["expert_placement"])
    ownership.write_text("changed")
    assert frozen.read_bytes() == original
    assert frozen.parent == config.parent
    with pytest.raises(ValueError, match="authentication"):
        prepare(tmp_path / "rejected", specification, source)
    assert not (tmp_path / "rejected").exists()
