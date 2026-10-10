# SPDX-License-Identifier: Apache-2.0
"""Byte-sealed bundle tests with fake host compiler outputs; no device runtime."""

import ast
import builtins
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp import build_streaming as builder
from tools.qwen4exp.streaming_protocol import canonical_sha256, seal_manifest

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "artifacts/qwen38-streaming-upgrade/T1/reference-v2.json"


@pytest.fixture
def source_root(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for relative in builder.REQUIRED:
        source, target = ROOT / relative, root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text("# Fake pending integration source for isolated CPU bundle tests.\n")
    sdk = json.loads((ROOT / "artifacts/qwen38-streaming-upgrade/T2/sdk-provenance.json").read_text())
    for name in sdk["files"]:
        shutil.copyfile(
            ROOT / "artifacts/qwen38-streaming-upgrade/T2" / name, root / "artifacts/qwen38-streaming-upgrade/T2" / name
        )
    original = subprocess.run

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="a" * 40 + "\n", returncode=0)
        return original(command, **kwargs)

    monkeypatch.setattr(builder.subprocess, "run", run)
    return root


def prepare(source, directory, version=1):
    return builder.prepare_bundle(
        directory, source, version, REFERENCE, {"schedule": {"chunk_tokens": 1024}, "offline_test": True}
    )


@pytest.fixture
def mock_compile(tmp_path, monkeypatch):
    calls = []
    original = builder.subprocess.run

    def run(command, **kwargs):
        if command[0] == "git":
            return original(command, **kwargs)
        calls.append(command)
        if len(command) == 2 and command[1] == "--probe":
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"landlock_abi": 8, "probe": "query_only"}) + "\n", stderr=""
            )
        wrapped = "--" in command
        child = command[command.index("--") + 1 :] if wrapped else command
        if child[0] == "c++":
            output = Path(child[child.index("-o") + 1])
            output.write_bytes(b"MOCK CPU compiler output\0" + " ".join(child).encode())
        else:
            output = Path(child[2])
            output.write_bytes(b"MOCK CPU native bytes\0" + Path(child[1]).read_bytes())
        attestation = {
            "containment": "landlock_seccomp",
            "landlock_abi": 8,
            "restricted": True,
            "nonstandard_fds_closed": True,
            "driver_ioctls_and_sockets_denied": True,
        }
        return SimpleNamespace(returncode=0, stdout="", stderr=json.dumps(attestation) + "\n" if wrapped else "")

    monkeypatch.setattr(builder.subprocess, "run", run)
    monkeypatch.setattr(
        builder,
        "_toolchain",
        lambda cann, **kwargs: {
            "cann": tmp_path / "sdk",
            "npu": tmp_path / "torch_npu_metadata",
            "torch_root": tmp_path / "torch_metadata",
            "abi": 1,
            "versions": {"torch": "CPU mock", "torch-npu": "not imported"},
            "byte_identities": [],
        },
    )
    return calls


def test_preparation_freezes_actual_sources_contract_config_and_native_closure(source_root, tmp_path):
    directory = tmp_path / "bundle"
    value = prepare(source_root, directory)
    assert value["namespace"] == "qwen_streaming_v1"
    assert value["configuration_sha256"] == canonical_sha256(value["configuration"])
    assert value["external_headers_requiring_host_compile"] == ["kernel_operator.h"]
    assert "tools/qwen4exp/qwen_streaming_operands.h" in value["native_include_closure"]
    assert value["admissible_for_loading"] is False
    assert builder.verify_bundle(directory) == value
    for entry in value["assets"]:
        assert builder.digest(directory / entry["path"]) == entry["sha256"]
    assert not (directory / "candidate.json").exists()


def test_source_edit_after_freeze_cannot_mutate_frozen_bundle(source_root, tmp_path):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    path = source_root / "tools/qwen4exp/native_streaming.cpp"
    path.write_text("broken later source")
    assert builder.verify_bundle(directory)["admissible_for_loading"] is False
    assert "qwen_streaming_projection_v1" in (directory / "sources/tools/qwen4exp/native_streaming.cpp").read_text()


@pytest.mark.parametrize("version", [0, -1, True, "1"])
def test_invalid_namespace_version_rejected(source_root, tmp_path, version):
    with pytest.raises(ValueError):
        prepare(source_root, tmp_path / "bundle", version)


def test_append_only_directory_rejected(source_root, tmp_path):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    with pytest.raises(FileExistsError):
        prepare(source_root, directory)


@pytest.mark.parametrize("bad", ["missing", "include", "entrypoint", "contract", "header", "symlink"])
def test_unresolved_or_inconsistent_dependencies_fail_before_directory_created(source_root, tmp_path, bad):
    target = source_root / "tools/qwen4exp/native_streaming.cpp"
    if bad == "missing":
        target.unlink()
    elif bad == "include":
        target.write_text('#include "missing_private_header.h"\n' + target.read_text())
    elif bad == "entrypoint":
        target.write_text(target.read_text().replace("qwen_streaming_columns_v1", "missing_column_entry"))
    elif bad == "contract":
        path = source_root / "artifacts/qwen38-streaming-upgrade/T2/contract.json"
        value = json.loads(path.read_text())
        value["contract_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    elif bad == "header":
        path = source_root / "tools/qwen4exp/qwen_streaming_contract.h"
        path.write_text("stale contract header")
    else:
        content = target.read_text()
        target.unlink()
        linked = tmp_path / "outside.cpp"
        linked.write_text(content)
        target.symlink_to(linked)
    directory = tmp_path / "bundle"
    with pytest.raises(ValueError):
        prepare(source_root, directory)
    assert not directory.exists()


@pytest.mark.parametrize("corruption", ["source", "prepared", "configuration", "escape"])
def test_prepared_verify_rejects_changed_bytes_or_seals(source_root, tmp_path, corruption):
    directory = tmp_path / "bundle"
    value = prepare(source_root, directory)
    if corruption == "source":
        (directory / value["assets"][0]["path"]).write_text("tampered")
    elif corruption == "configuration":
        (directory / "configuration.json").write_text("{}")
    else:
        value["namespace"] = "wrong"
        if corruption == "escape":
            value["assets"][0]["path"] = "../outside"
            value = seal_manifest(value)
        (directory / "prepared.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        builder.verify_bundle(directory)


def test_all_resources_host_commands_use_frozen_sources_unique_namespace_and_relative_inventory(
    source_root, tmp_path, mock_compile
):
    directory = tmp_path / "bundle"
    prepared = prepare(source_root, directory, 7)
    candidate = builder.build_bundle(directory, tmp_path / "sdk")
    assert candidate["namespace"] == "qwen_streaming_v7"
    assert candidate["configuration"] == prepared["configuration"]
    assert candidate["configuration_sha256"] == prepared["configuration_sha256"]
    assert len(candidate["binaries"]) == 4 and len(candidate["bridges"]) == 1
    assert sum(len(entry["entrypoints"]) for entry in candidate["binaries"]) == 6
    assert candidate["resources_sha256"] == canonical_sha256(builder.resource_inventory(candidate))
    assert candidate["hardware_validated"] is False
    assert not candidate["admissible_without_gate_evidence"]
    assert builder.verify_bundle(directory, require_compiled=True) == candidate
    assert len(mock_compile) == 8
    assert str(directory / "sources/tools/qwen4exp/host_compile_sandbox.cpp") in mock_compile[0]
    assert "-static" in mock_compile[0]
    assert str(directory / "sources/tools/qwen4exp/compile_streaming.cpp") in mock_compile[2]
    assert not any("compile_reconstruction.cpp" in str(argument) for command in mock_compile for argument in command)
    assert all(
        any(str(directory / "sources") in str(argument) for argument in command)
        for command in mock_compile
        if command[-1] != "--probe"
    )
    assert any("-DGLM_RECONSTRUCTION_NAMESPACE=qwen_streaming_v7" in command for command in mock_compile)
    receipt = json.loads((directory / "build-receipt.json").read_text())
    assert receipt["target"] == "dav-2002"
    for key in ("npu_opened", "library_loaded", "kernels_launched", "server_contacted"):
        assert receipt[key] is False


@pytest.mark.parametrize("corruption", ["binary", "bridge", "receipt", "inventory", "entrypoints", "configuration"])
def test_compiled_verify_rehashes_actual_bytes_and_binding(source_root, tmp_path, mock_compile, corruption):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    candidate = builder.build_bundle(directory, tmp_path / "sdk")
    if corruption in ("binary", "bridge"):
        field = "binaries" if corruption == "binary" else "bridges"
        (directory / candidate[field][0]["path"]).write_bytes(b"changed native bytes")
    elif corruption == "receipt":
        (directory / "build-receipt.json").write_text("{}")
    else:
        if corruption == "inventory":
            candidate["resources_sha256"] = "0" * 64
        elif corruption == "entrypoints":
            candidate["binaries"][0]["entrypoints"] = []
            candidate["resources_sha256"] = canonical_sha256(builder.resource_inventory(candidate))
        else:
            candidate["configuration"] = {"changed": True}
        candidate.pop("candidate_sha256")
        candidate["candidate_sha256"] = canonical_sha256(candidate)
        (directory / "candidate.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        builder.verify_bundle(directory, require_compiled=True)


def test_no_rebuild_and_no_implicit_manifest_factory(source_root, tmp_path, mock_compile):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    builder.build_bundle(directory, tmp_path / "sdk")
    with pytest.raises(FileExistsError):
        builder.build_bundle(directory, tmp_path / "sdk")
    with pytest.raises(ValueError):
        builder.make_native_manifest(directory, "")
    source = "def prepare():\n    raise RuntimeError('admission required')\n"
    manifest = builder.make_native_manifest(directory, source)
    assert manifest["validation_source"] == source
    assert manifest["operators"] == ["qwen_streaming_v1::launch"]
    assert all(Path(entry["path"]).is_file() for entry in manifest["assets"] + manifest["libraries"])


def test_prepare_and_verifier_never_import_device_runtime(source_root, tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("torch_npu", "torch"):
            raise AssertionError("source preparation must not import tensor/device runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    builder.verify_bundle(directory)


def test_builder_has_no_device_load_or_execution_calls():
    tree = ast.parse((ROOT / "tools/qwen4exp/build_streaming.py").read_text())
    forbidden = {"load_library", "CDLL", "Kernel", "npu", "synchronize", "set_device", "launch", "aclrtSetDevice"}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in forbidden]
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name in ("torch_npu", "torch") for alias in node.names)
    ]


def test_contract_scalar_change_cannot_hide_behind_old_header_hash(source_root, tmp_path):
    path = source_root / "tools/qwen4exp/qwen_streaming_contract.h"
    path.write_text(path.read_text().replace("constexpr uint32_t M = 16;", "constexpr uint32_t M = 32;"))
    with pytest.raises(ValueError, match="constant mismatch"):
        prepare(source_root, tmp_path / "bundle")


def test_sdk_snapshot_change_rejected(source_root, tmp_path):
    sdk_root = source_root / "artifacts/qwen38-streaming-upgrade/T2"
    sdk = json.loads((sdk_root / "sdk-provenance.json").read_text())
    (sdk_root / next(iter(sdk["files"]))).write_text("changed header")
    with pytest.raises(ValueError, match="SDK snapshot"):
        prepare(source_root, tmp_path / "bundle")


@pytest.fixture
def toolchain_metadata(tmp_path, monkeypatch):
    cann, npu, torch_root = (tmp_path / value for value in ("sdk", "npu_metadata", "torch_metadata"))
    for path in (
        cann / "include/acl/acl.h",
        cann / "include/acl/acl_rt_compile.h",
        cann / "include/kernel_operator.h",
        cann / "lib64/libacl_rtc.so",
        cann / "lib64/libascendcl.so",
        npu / "include/torch_npu/csrc/core/npu/NPUStream.h",
        npu / "lib/libtorch_npu.so",
        torch_root / "share/cmake/Torch/TorchConfig.cmake",
        torch_root / "include/fixture.h",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CPU metadata fixture only\n")
    monkeypatch.setattr(
        builder.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str((npu if name == "torch_npu" else torch_root) / "__init__.py")),
    )
    monkeypatch.setattr(
        builder.importlib.metadata, "version", lambda name: "2.13.0+cpu" if name == "torch" else "2.13.0rc1"
    )
    evidence = tmp_path / "abi.json"
    value = {
        "torch_version": "2.13.0+cpu",
        "torch_abi": 1,
        "torch_npu_imported": False,
        "device_backend_autoload": "0",
        "npu_opened": False,
        "source": "guarded_cpu_torch_metadata",
    }
    evidence.write_text(json.dumps(value))
    return cann, torch_root, evidence, value


def test_missing_cmake_abi_requires_explicit_host_metadata_evidence(toolchain_metadata):
    cann, _, evidence, _ = toolchain_metadata
    with pytest.raises(RuntimeError, match="explicit ABI"):
        builder._toolchain(cann)
    value = builder._toolchain(cann, torch_abi=1, abi_evidence=evidence)
    assert value["abi"] == 1 and value["abi_evidence_path"] == evidence
    assert value["byte_identities"]


@pytest.mark.parametrize(
    "changed", ["torch_version", "torch_abi", "torch_npu_imported", "npu_opened", "device_backend_autoload", "source"]
)
def test_unproven_or_mismatched_abi_evidence_rejected(toolchain_metadata, changed):
    cann, _, evidence, value = toolchain_metadata
    value[changed] = True if changed in ("torch_npu_imported", "npu_opened") else "invalid"
    evidence.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="evidence"):
        builder._toolchain(cann, torch_abi=1, abi_evidence=evidence)


def test_cmake_abi_uses_metadata_only_and_rejects_conflicting_override(toolchain_metadata, monkeypatch):
    cann, torch_root, _, _ = toolchain_metadata
    (torch_root / "share/cmake/Torch/TorchConfig.cmake").write_text(
        'set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")\n'
    )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("torch", "torch_npu"):
            raise AssertionError("builder must not import tensor runtimes")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert builder._toolchain(cann)["abi"] == 0
    with pytest.raises(RuntimeError, match="conflicts"):
        builder._toolchain(cann, torch_abi=1)


@pytest.mark.parametrize("abi", [0, 1, 2])
def test_unsupported_containment_abi_cannot_start_sdk_compilers(source_root, tmp_path, mock_compile, monkeypatch, abi):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    original = builder.subprocess.run

    def run(command, **kwargs):
        if len(command) == 2 and command[1] == "--probe":
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"landlock_abi": abi, "probe": "query_only"}), stderr=""
            )
        return original(command, **kwargs)

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Landlock unavailable"):
        builder.build_bundle(directory, tmp_path / "sdk")
    assert len(mock_compile) == 1
    assert not (directory / "candidate.json").exists()


@pytest.mark.parametrize("failure", ["missing", "not_restricted", "socket_not_denied", "wrong_abi"])
def test_child_without_enforced_containment_cannot_publish_candidate(
    source_root, tmp_path, mock_compile, monkeypatch, failure
):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    original = builder.subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        if "--" in command:
            if failure == "missing":
                result.stderr = "no policy attestation\n"
            else:
                value = json.loads(result.stderr)
                if failure == "not_restricted":
                    value["restricted"] = False
                elif failure == "socket_not_denied":
                    value["driver_ioctls_and_sockets_denied"] = False
                else:
                    value["landlock_abi"] = 7
                result.stderr = json.dumps(value)
        return result

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="containment"):
        builder.build_bundle(directory, tmp_path / "sdk")
    assert not (directory / "candidate.json").exists()


@pytest.mark.parametrize("corruption", ["old_receipt", "helper_bytes", "log_bytes", "dev_allowance"])
def test_compiled_receipt_requires_actual_containment_bytes_and_safe_policy(
    source_root, tmp_path, mock_compile, corruption
):
    directory = tmp_path / "bundle"
    prepare(source_root, directory)
    candidate = builder.build_bundle(directory, tmp_path / "sdk")
    receipt_path = directory / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if corruption == "helper_bytes":
        (directory / receipt["filesystem_containment"]["helper"]["path"]).write_bytes(b"different helper")
    elif corruption == "log_bytes":
        (directory / receipt["command_attestations"][0]["stderr"]["path"]).write_text("changed attestation log")
    else:
        if corruption == "old_receipt":
            del receipt["filesystem_containment"]
        else:
            policy = receipt["filesystem_containment"]["policy"]
            policy["allow_read"] = sorted(set(policy["allow_read"] + ["/dev"]))
            receipt["filesystem_containment"]["policy_sha256"] = canonical_sha256(policy)
        receipt = seal_manifest(receipt)
        receipt_path.write_text(json.dumps(receipt))
        candidate["build_receipt"]["sha256"] = builder.digest(receipt_path)
        candidate.pop("candidate_sha256")
        candidate["candidate_sha256"] = canonical_sha256(candidate)
        (directory / "candidate.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        builder.verify_bundle(directory, require_compiled=True)


def test_loader_injection_removed_without_mutating_parent_environment(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "unsafe.so")
    monkeypatch.setenv("LD_AUDIT", "unsafe-audit.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/explicit/sdk/lib")
    environment = builder._compiler_environment()
    assert "LD_PRELOAD" not in environment and "LD_AUDIT" not in environment
    assert environment["LD_LIBRARY_PATH"] == "/explicit/sdk/lib"
    assert builder.os.environ["LD_PRELOAD"] == "unsafe.so"


def test_next_profile_freezes_new_contract_and_only_projection_resource(source_root, tmp_path, mock_compile):
    for relative in builder.NEXT_REQUIRED:
        target = source_root / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(ROOT / relative, target)
    directory = tmp_path / "next"
    builder.prepare_bundle(
        directory, source_root, 6, REFERENCE, {"projection_variant": "m32n160_v2", "layers": {"native_wy": False}}
    )
    prepared = builder.verify_bundle(directory)
    assert (
        prepared["contract_sha256"]
        != json.loads((ROOT / "artifacts/qwen38-streaming-upgrade/T2/contract.json").read_text())["contract_sha256"]
    )
    candidate = builder.build_bundle(directory, "unused")
    assert candidate["native_components"] == ["native_streaming_next", "native_route_gather"]
    projection = next(b for b in candidate["binaries"] if b["path"] == "binaries/native_streaming_next.bin")
    assert projection["entrypoints"] == ["qwen_streaming_projection_v2", "qwen_streaming_columns_v2"]
    assert builder.verify_bundle(directory, require_compiled=True) == candidate


@pytest.mark.parametrize(
    "config", [{"projection_variant": "unknown"}, {"projection_variant": "m32n160_v2", "layers": {"native_wy": True}}]
)
def test_next_profile_rejects_unknown_variant_and_failed_wy(config):
    with pytest.raises(ValueError):
        builder.projection_profile(config)


def test_contained_compiler_has_explicit_transitive_library_paths_without_parent_mutation(monkeypatch, tmp_path):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unused/parent/path")
    env = builder._compiler_environment((tmp_path / "sdk/lib64", tmp_path / "torch/lib"))
    assert env["LD_LIBRARY_PATH"] == f"{tmp_path}/sdk/lib64:{tmp_path}/torch/lib:/unused/parent/path"
    assert os.environ["LD_LIBRARY_PATH"] == "/unused/parent/path"
