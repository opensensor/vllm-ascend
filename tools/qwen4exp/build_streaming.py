# SPDX-License-Identifier: Apache-2.0
"""Append-only source/binary bundle preparation and host-only CANN compilation.

Preparation imports no device runtime. Compilation never loads its resulting
library or opens a device. Only a separately admitted controller may load it.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path, PurePosixPath

import regex as re

from tools.qwen4exp.streaming_protocol import SOURCE_ROOTS, canonical_sha256, seal_manifest, validate_manifest

SCHEMA_VERSION = 1
MIN_LANDLOCK_ABI = 3
KERNELS = (
    ("native_streaming", ("qwen_streaming_projection_v1", "qwen_streaming_columns_v1")),
    ("native_wy", ("qwen_fused_wy_v1",)),
    ("native_state_layout", ("qwen_state_gather_v1", "qwen_state_scatter_v1")),
    ("native_route_gather", ("qwen_local_route_gather_v1",)),
)
REQUIRED = (
    "tools/qwen4exp/streaming_protocol.py",
    "tools/qwen4exp/streaming_memory.py",
    "tools/qwen4exp/streaming_operands.py",
    "tools/qwen4exp/streaming_epilogue.py",
    "tools/qwen4exp/streaming_schedule.py",
    "tools/qwen4exp/streaming_layer.py",
    "tools/qwen4exp/streaming_candidate.py",
    "tools/qwen4exp/streaming_resident.py",
    "tools/qwen4exp/resident_candidates/streaming.py",
    "tools/qwen4exp/native_streaming.py",
    "tools/qwen4exp/native_prefill.py",
    "tools/qwen4exp/native_state_layout.py",
    "tools/qwen4exp/build_streaming.py",
    "tools/qwen4exp/qwen_streaming_contract.h",
    "tools/qwen4exp/qwen_streaming_operands.h",
    "tools/qwen4exp/qwen_streaming_projection.h",
    "tools/qwen4exp/qwen_streaming_epilogue.h",
    "tools/qwen4exp/compile_streaming.cpp",
    "tools/qwen4exp/host_compile_sandbox.cpp",
    "tools/glm_perf/reconstruction_bridge.cpp",
    "artifacts/qwen38-streaming-upgrade/T2/contract.json",
    "artifacts/qwen38-streaming-upgrade/T2/sdk-provenance.json",
    "vllm_ascend/models/qwen4_exp/model.py",
    "vllm_ascend/models/qwen4_exp/w4_moe.py",
    "vllm_ascend/models/qwen4_exp/w4a8_int4.py",
    "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py",
    *(f"tools/qwen4exp/{name}.cpp" for name, _ in KERNELS),
)
INCLUDES = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)
EXTERNAL_HEADERS = ("kernel_operator.h",)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _path(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("unsafe bundle path")
    path = root.joinpath(*value.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("bundle path escapes its root")
    return path


def _write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def resource_inventory(candidate):
    """Canonical byte inventory shared by the admitted loader; no circular SHA."""
    return {
        "namespace": candidate["namespace"],
        "binaries": sorted(candidate["binaries"], key=lambda item: item["path"]),
        "bridges": sorted(candidate["bridges"], key=lambda item: item["path"]),
    }


def _source_files(root, extra_assets):
    paths = set(REQUIRED) | set(extra_assets)
    sdk_path = root / "artifacts/qwen38-streaming-upgrade/T2/sdk-provenance.json"
    if not sdk_path.is_file():
        raise ValueError("SDK byte provenance missing")
    for name, expected in json.loads(sdk_path.read_text())["files"].items():
        path = _path(sdk_path.parent, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("SDK snapshot dependency mismatch")
        paths.add(path.relative_to(root).as_posix())
    for relative in (*SOURCE_ROOTS, "vllm_ascend/", "tools/glm_perf/"):
        path = _path(root, relative)
        if path.is_dir():
            paths.update(
                item.relative_to(root).as_posix()
                for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix not in (".pyc", ".bin", ".so")
            )
        elif path.is_file():
            paths.add(relative)
    for relative in sorted(paths):
        path = _path(root, relative)
        if not path.is_file():
            raise ValueError(f"required source dependency missing: {relative}")
        if path.is_symlink():
            raise ValueError(f"source symlink must be resolved explicitly: {relative}")
    return sorted(paths)


def _native_closure(root):
    """Resolve native quoted includes; SDK-owned kernel headers are explicit."""
    pending = [root / f"tools/qwen4exp/{name}.cpp" for name, _ in KERNELS]
    visited, external = set(), set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for name in INCLUDES.findall(path.read_text()):
            if name in EXTERNAL_HEADERS:
                external.add(name)
                continue
            target = path.parent / name
            if not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"unresolved native source include: {name}")
            pending.append(target)
    for name, entrypoints in KERNELS:
        source = (root / f"tools/qwen4exp/{name}.cpp").read_text()
        if any(not re.search(r"\bvoid\s+" + re.escape(entry) + r"\s*\(", source) for entry in entrypoints):
            raise ValueError("native source entrypoint missing")
    return sorted(path.relative_to(root).as_posix() for path in visited), sorted(external)


def _verify_contract(contract, header):
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if contract.get("contract_sha256") != canonical_sha256(body):
        raise ValueError("streaming contract seal mismatch")
    if f'CONTRACT_SHA256 = "{contract["contract_sha256"]}"' not in header:
        raise ValueError("streaming contract header mismatch")
    constants = dict(contract["constants"], ABI_VERSION=contract["abi_version"])
    for region in contract["regions"]:
        constants[region["name"].upper() + "_OFFSET"] = region["offset"]
        constants[region["name"].upper() + "_BYTES"] = region["nbytes"]
    for name, expected in constants.items():
        match = re.search(r"constexpr\s+uint32_t\s+" + re.escape(name) + r"\s*=\s*(\d+)\s*;", header)
        if match is None or int(match.group(1)) != expected:
            raise ValueError("streaming contract header constant mismatch")


def prepare_bundle(directory, source_root, version, reference_manifest, configuration, extra_assets=()):
    """Freeze complete source scope and identities; prepared bundles cannot serve."""
    root = Path(source_root).resolve(strict=True)
    directory = Path(directory).resolve()
    if type(version) is not int or version <= 0 or not isinstance(configuration, dict) or not configuration:
        raise ValueError("positive namespace version and complete configuration snapshot required")
    configuration_sha256 = canonical_sha256(configuration)
    reference = json.loads(Path(reference_manifest).read_text())
    validate_manifest(reference)
    files = _source_files(root, extra_assets)
    closure, external = _native_closure(root)
    if not set(closure) <= set(files):
        raise ValueError("native include closure omitted from source inventory")
    contract = json.loads((root / "artifacts/qwen38-streaming-upgrade/T2/contract.json").read_text())
    header = (root / "tools/qwen4exp/qwen_streaming_contract.h").read_text()
    _verify_contract(contract, header)
    directory.mkdir(parents=True, exist_ok=False)
    assets = []
    for relative in files:
        source = _path(root, relative)
        target = _path(directory, "sources/" + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        actual = digest(target)
        if actual != digest(source):
            raise ValueError("source changed during freeze")
        role = "source"
        if relative.endswith("T2/contract.json"):
            role = "streaming_contract"
        elif relative.endswith("qwen_streaming_contract.h"):
            role = "streaming_contract_header"
        assets.append({"path": target.relative_to(directory).as_posix(), "sha256": actual, "role": role})
    _write_new(directory / "configuration.json", configuration)
    assets.append(
        {"path": "configuration.json", "sha256": digest(directory / "configuration.json"), "role": "configuration"}
    )
    frozen_reference = directory / "reference.json"
    _write_new(frozen_reference, reference)
    assets.append({"path": "reference.json", "sha256": digest(frozen_reference), "role": "baseline_reference"})
    source_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    prepared = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "qwen_streaming_prepared_bundle",
            "namespace": f"qwen_streaming_v{version}",
            "version": version,
            "base_reference_sha256": reference["manifest_sha256"],
            "configuration_sha256": configuration_sha256,
            "configuration": configuration,
            "contract_sha256": contract["contract_sha256"],
            "source_commit": source_commit,
            "baseline_gitlinks": reference["source"].get("gitlinks", []),
            "assets": sorted(assets, key=lambda item: item["path"]),
            "native_include_closure": closure,
            "external_headers_requiring_host_compile": external,
            "hardware_validated": False,
            "server_modified": False,
            "admissible_for_loading": False,
        }
    )
    _write_new(directory / "prepared.json", prepared)
    verify_bundle(directory)
    return prepared


def _verify_assets(directory, entries):
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("duplicate bundle asset path")
    for entry in entries:
        path = _path(directory, entry["path"])
        if not _sha(entry["sha256"]) or not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"bundle file-byte mismatch: {entry['path']}")


def verify_bundle(directory, require_compiled=False):
    """Rehash source and compiled bytes without importing/loading any code."""
    directory = Path(directory).resolve(strict=True)
    prepared = json.loads((directory / "prepared.json").read_text())
    if prepared != seal_manifest(prepared) or prepared.get("kind") != "qwen_streaming_prepared_bundle":
        raise ValueError("prepared bundle seal mismatch")
    _verify_assets(directory, prepared["assets"])
    expected_assets = {"sources/" + path for path in REQUIRED} | {"configuration.json", "reference.json"}
    if not expected_assets <= {entry["path"] for entry in prepared["assets"]}:
        raise ValueError("prepared source inventory incomplete")
    if (
        type(prepared.get("version")) is not int
        or prepared["version"] <= 0
        or prepared["namespace"] != f"qwen_streaming_v{prepared['version']}"
    ):
        raise ValueError("prepared namespace identity mismatch")
    closure, external = _native_closure(directory / "sources")
    if closure != prepared["native_include_closure"] or external != prepared["external_headers_requiring_host_compile"]:
        raise ValueError("frozen native closure mismatch")
    contract = json.loads((directory / "sources/artifacts/qwen38-streaming-upgrade/T2/contract.json").read_text())
    _verify_contract(contract, (directory / "sources/tools/qwen4exp/qwen_streaming_contract.h").read_text())
    if contract["contract_sha256"] != prepared["contract_sha256"]:
        raise ValueError("prepared logical contract identity mismatch")
    configuration = json.loads((directory / "configuration.json").read_text())
    if (
        prepared.get("configuration") != configuration
        or canonical_sha256(configuration) != prepared["configuration_sha256"]
    ):
        raise ValueError("configuration snapshot mismatch")
    if not require_compiled:
        return prepared
    candidate = json.loads((directory / "candidate.json").read_text())
    body = {key: value for key, value in candidate.items() if key != "candidate_sha256"}
    if candidate.get("candidate_sha256") != canonical_sha256(body):
        raise ValueError("candidate seal mismatch")
    for field in (
        "namespace",
        "base_reference_sha256",
        "configuration_sha256",
        "configuration",
        "contract_sha256",
        "assets",
    ):
        if candidate[field] != prepared[field]:
            raise ValueError("candidate differs from prepared source/configuration")
    if candidate["resources_sha256"] != canonical_sha256(resource_inventory(candidate)):
        raise ValueError("native resource inventory mismatch")
    _verify_assets(directory, candidate["assets"] + candidate["binaries"] + candidate["bridges"])
    expected_binaries = {"binaries/" + name + ".bin": list(entries) for name, entries in KERNELS}
    if {entry["path"]: entry["entrypoints"] for entry in candidate["binaries"]} != expected_binaries:
        raise ValueError("native per-binary entrypoint identity mismatch")
    if candidate["native_components"] != [name for name, _ in KERNELS]:
        raise ValueError("native component coverage mismatch")
    if len(candidate["bridges"]) != 1 or candidate["bridges"][0]["path"] != f"bridge/{candidate['namespace']}.so":
        raise ValueError("versioned bridge identity mismatch")
    expected = {entry for _, entries in KERNELS for entry in entries}
    actual = [entry for binary in candidate["binaries"] for entry in binary["entrypoints"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("native entrypoint coverage mismatch")
    receipt_entry = candidate["build_receipt"]
    _verify_assets(directory, [receipt_entry])
    receipt = json.loads(_path(directory, receipt_entry["path"]).read_text())
    if (
        receipt != seal_manifest(receipt)
        or receipt.get("kind") != "host_only_compile_receipt"
        or receipt.get("target") != "dav-2002"
        or receipt.get("prepared_manifest_sha256") != prepared["manifest_sha256"]
        or receipt.get("resources_sha256") != candidate["resources_sha256"]
        or any(
            receipt.get(key) is not False
            for key in ("npu_opened", "library_loaded", "kernels_launched", "server_contacted")
        )
    ):
        raise ValueError("host compilation receipt mismatch")
    containment = receipt.get("filesystem_containment", {})
    _verify_containment(directory, containment, prepared)
    attestations = receipt.get("command_attestations", [])
    if len(attestations) != len(KERNELS) + 2:
        raise ValueError("complete host compilation containment attestations required")
    for record in attestations:
        _verify_assets(directory, [record["stdout"], record["stderr"]])
        attestation = record["attestation"]
        if (
            record["command_sha256"] != canonical_sha256(record["command"])
            or record["prefix_sha256"] != canonical_sha256(containment["command_prefix"])
            or attestation.get("containment") != "landlock_seccomp"
            or attestation.get("landlock_abi") != containment["abi"]
            or any(
                attestation.get(key) is not True
                for key in ("restricted", "nonstandard_fds_closed", "driver_ioctls_and_sockets_denied")
            )
        ):
            raise ValueError("host command containment receipt mismatch")
        stderr = _path(directory, record["stderr"]["path"]).read_text()
        found = False
        for line in stderr.splitlines():
            try:
                found = found or json.loads(line) == attestation
            except ValueError:
                continue
        if not found:
            raise ValueError("host command containment attestation absent from log")
    if receipt.get("abi_evidence") is not None:
        _verify_assets(directory, [receipt["abi_evidence"]])
    return candidate


def _toolchain(cann, torch_abi=None, abi_evidence=None):
    """Discover matching headers/libraries using metadata; no torch-npu import."""
    cann = Path(cann).resolve(strict=True)
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("matching torch-npu package metadata required")
    npu = Path(spec.origin).parent
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError("matching torch package metadata required")
    torch_root = Path(torch_spec.origin).parent
    required = [
        cann / "include/acl/acl.h",
        cann / "include/acl/acl_rt_compile.h",
        cann / "lib64/libacl_rtc.so",
        cann / "lib64/libascendcl.so",
        npu / "include/torch_npu/csrc/core/npu/NPUStream.h",
        npu / "lib/libtorch_npu.so",
    ]
    headers = sorted(cann.rglob("kernel_operator.h"))
    if not headers or any(not path.is_file() for path in required):
        raise RuntimeError("unresolved host compilation toolchain dependency")
    header_files = sorted(
        {
            path
            for header_root in (
                cann / "include",
                *(path.parent for path in headers),
                torch_root / "include",
                npu / "include",
            )
            for path in header_root.rglob("*")
            if path.is_file() and path.suffix in (".h", ".hpp", ".inc", ".cuh")
        }
    )
    # Read ABI from installed CMake metadata. Importing Torch itself could
    # autoload a device backend, so neither Torch nor torch-npu is imported.
    cmake = torch_root / "share/cmake/Torch/TorchConfig.cmake"
    if not cmake.is_file():
        raise RuntimeError("Torch CMake ABI metadata missing")
    abi_matches = set(re.findall(r"_GLIBCXX_USE_CXX11_ABI=([01])", cmake.read_text()))
    evidence_path = None
    if len(abi_matches) == 1:
        abi = int(next(iter(abi_matches)))
        if torch_abi is not None and (type(torch_abi) is not int or torch_abi != abi):
            raise RuntimeError("explicit Torch ABI conflicts with CMake metadata")
    else:
        if abi_matches or type(torch_abi) is not int or torch_abi not in (0, 1) or abi_evidence is None:
            raise RuntimeError("Torch ABI metadata missing; explicit ABI and guarded CPU metadata evidence required")
        evidence_path = Path(abi_evidence).resolve(strict=True)
        evidence = json.loads(evidence_path.read_text())
        if (
            evidence.get("torch_abi") != torch_abi
            or evidence.get("torch_version") != importlib.metadata.version("torch")
            or evidence.get("torch_npu_imported") is not False
            or evidence.get("npu_opened") is not False
            or evidence.get("device_backend_autoload") != "0"
            or evidence.get("source") != "guarded_cpu_torch_metadata"
        ):
            raise RuntimeError("Torch ABI evidence does not establish a matching host-only query")
        abi = torch_abi
    return {
        "cann": cann,
        "npu": npu,
        "torch_root": torch_root,
        "abi": abi,
        "abi_evidence_path": evidence_path,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu")},
        "byte_identities": [
            {"path": str(path.resolve()), "sha256": digest(path)} for path in (*required, cmake, *header_files)
        ],
    }


def _compiler_environment():
    """Prevent loader injection before the unlinked containment helper starts."""
    environment = dict(os.environ)
    for name in ("LD_PRELOAD", "LD_AUDIT"):
        environment.pop(name, None)
    return environment


def _build_containment(directory, source_root):
    helper = directory / "build/host-compile-sandbox"
    source = source_root / "tools/qwen4exp/host_compile_sandbox.cpp"
    subprocess.run(
        ["c++", "-static", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(helper)],
        check=True,
        env=_compiler_environment(),
    )
    probe = subprocess.run(
        [str(helper), "--probe"], check=True, capture_output=True, text=True, env=_compiler_environment()
    )
    try:
        query = json.loads(probe.stdout)
        abi = query["landlock_abi"]
        if query.get("probe") != "query_only" or type(abi) is not int:
            raise ValueError("invalid Landlock ABI query schema")
    except (ValueError, AttributeError, KeyError, TypeError) as error:
        raise RuntimeError("Landlock helper returned no verified ABI") from error
    if abi < MIN_LANDLOCK_ABI:
        raise RuntimeError("Landlock unavailable; host compilation denied")
    return helper, abi, probe.stdout, digest(source)


def _containment_policy(directory, toolchain, helper, abi, probe_stdout, source_sha256):
    safe_devices = ["/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"]
    read = sorted(
        {
            str(Path(path).resolve())
            for path in (
                "/usr",
                "/lib",
                "/lib64",
                "/etc",
                "/proc",
                directory,
                toolchain["cann"],
                toolchain["npu"],
                toolchain["torch_root"],
                *safe_devices,
            )
        }
    )
    write = sorted({str(Path(path).resolve()) for path in (directory, "/tmp", *safe_devices)})
    policy = {
        "allow_read": read,
        "allow_write": write,
        "safe_device_files": safe_devices,
        "deny_all_other_device_paths": True,
        "close_inherited_fds": True,
        "no_new_privs": True,
        "loader_injection_removed": ["LD_PRELOAD", "LD_AUDIT"],
        "driver_ioctls_and_sockets_denied": True,
    }
    prefix = [str(helper)]
    for name, paths in (("--allow-read", read), ("--allow-write", write)):
        for path in paths:
            prefix.extend((name, path))
    prefix.append("--")
    return {
        "mechanism": "landlock_seccomp",
        "abi": abi,
        "helper": {"path": helper.relative_to(directory).as_posix(), "sha256": digest(helper)},
        "helper_source_sha256": source_sha256,
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "probe": {"argv": [str(helper), "--probe"], "stdout": probe_stdout, "returncode": 0},
        "command_prefix": prefix,
        "verified_before_commands": True,
        "verification_basis": "kernel_landlock_filesystem_denial",
    }


def _verify_containment(directory, containment, prepared):
    if (
        containment.get("mechanism") != "landlock_seccomp"
        or type(containment.get("abi")) is not int
        or containment["abi"] < MIN_LANDLOCK_ABI
        or containment.get("verified_before_commands") is not True
        or containment.get("verification_basis") != "kernel_landlock_filesystem_denial"
    ):
        raise ValueError("verified filesystem containment required")
    _verify_assets(directory, [containment["helper"]])
    source_path = "sources/tools/qwen4exp/host_compile_sandbox.cpp"
    source = next((entry for entry in prepared["assets"] if entry["path"] == source_path), None)
    if source is None or containment["helper_source_sha256"] != source["sha256"]:
        raise ValueError("containment helper source mismatch")
    policy = containment["policy"]
    if (
        canonical_sha256(policy) != containment.get("policy_sha256")
        or any(
            policy.get(key) is not True
            for key in (
                "deny_all_other_device_paths",
                "close_inherited_fds",
                "no_new_privs",
                "driver_ioctls_and_sockets_denied",
            )
        )
        or policy.get("safe_device_files") != ["/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"]
        or policy.get("loader_injection_removed") != ["LD_PRELOAD", "LD_AUDIT"]
    ):
        raise ValueError("filesystem containment policy mismatch")
    for paths in (policy["allow_read"], policy["allow_write"]):
        if paths != sorted(set(paths)):
            raise ValueError("containment paths must be sorted and unique")
        for value in paths:
            path = Path(value)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or value in ("/", "/dev", "/srv", "/home", "/root", "/run", "/var")
            ):
                raise ValueError("unsafe containment path allowance")
            if path.is_relative_to("/dev") and value not in policy["safe_device_files"]:
                raise ValueError("driver device allowance forbidden")
    probe = containment["probe"]
    try:
        query = json.loads(probe.get("stdout", ""))
    except (ValueError, TypeError) as error:
        raise ValueError("containment ABI probe mismatch") from error
    if probe.get("returncode") != 0 or query != {"landlock_abi": containment["abi"], "probe": "query_only"}:
        raise ValueError("containment ABI probe mismatch")
    prefix = containment["command_prefix"]
    if not prefix or prefix[0] != probe["argv"][0] or prefix[-1] != "--":
        raise ValueError("containment command prefix mismatch")
    expected = [prefix[0]]
    for name, paths in (("--allow-read", policy["allow_read"]), ("--allow-write", policy["allow_write"])):
        for path in paths:
            expected.extend((name, path))
    expected.append("--")
    if prefix != expected or probe["argv"] != [prefix[0], "--probe"]:
        raise ValueError("containment command allowance mismatch")


def build_bundle(directory, cann_root, torch_abi=None, abi_evidence=None):
    """Compile all coherent resources from frozen sources; never load outputs."""
    directory = Path(directory).resolve(strict=True)
    prepared = verify_bundle(directory)
    if (directory / "candidate.json").exists() or (directory / "build-receipt.json").exists():
        raise FileExistsError("compiled bundle is append-only; prepare a new version/directory")
    build_dir = directory / "build"
    build_dir.mkdir(exist_ok=False)
    helper, landlock_abi, probe_stdout, helper_source_sha = _build_containment(directory, directory / "sources")
    toolchain = _toolchain(cann_root, torch_abi=torch_abi, abi_evidence=abi_evidence)
    containment = _containment_policy(directory, toolchain, helper, landlock_abi, probe_stdout, helper_source_sha)
    _verify_containment(directory, containment, prepared)

    command_attestations = []

    def run_compiler(command):
        result = subprocess.run(
            [*containment["command_prefix"], *command], capture_output=True, text=True, env=_compiler_environment()
        )
        index = len(command_attestations)
        logs = {}
        for name in ("stdout", "stderr"):
            path = build_dir / f"command-{index}.{name}.log"
            with path.open("x") as stream:
                stream.write(getattr(result, name))
            logs[name] = {"path": path.relative_to(directory).as_posix(), "sha256": digest(path)}
        attestations = []
        for line in result.stderr.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("containment") == "landlock_seccomp":
                attestations.append(value)
        if len(attestations) != 1:
            raise RuntimeError("compiler child lacks unique enforced-containment attestation")
        value = attestations[0]
        if value.get("landlock_abi") != landlock_abi or any(
            value.get(key) is not True
            for key in ("restricted", "nonstandard_fds_closed", "driver_ioctls_and_sockets_denied")
        ):
            raise RuntimeError("compiler child containment attestation mismatch")
        command_attestations.append(
            {
                "command": command,
                "command_sha256": canonical_sha256(command),
                "prefix_sha256": canonical_sha256(containment["command_prefix"]),
                "attestation": value,
                **logs,
            }
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)

    frozen_abi = None
    if toolchain.get("abi_evidence_path") is not None:
        path = build_dir / "torch-abi-evidence.json"
        shutil.copyfile(toolchain["abi_evidence_path"], path)
        frozen_abi = {"path": path.relative_to(directory).as_posix(), "sha256": digest(path)}
    cann, npu, torch_root = (toolchain[key] for key in ("cann", "npu", "torch_root"))
    root = directory / "sources"
    compiler = build_dir / "compile-streaming"
    run_compiler(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(root / "tools/qwen4exp/compile_streaming.cpp"),
            f"-I{cann / 'include'}",
            f"-L{cann / 'lib64'}",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-o",
            str(compiler),
        ],
    )
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    options = [
        "--npu-arch=dav-2002",
        f"-I{root / 'tools/qwen4exp'}",
        f"--sysroot={cann / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
    ]
    binaries = []
    (directory / "binaries").mkdir(exist_ok=False)
    for name, entrypoints in KERNELS:
        output = directory / "binaries" / f"{name}.bin"
        run_compiler([str(compiler), str(root / f"tools/qwen4exp/{name}.cpp"), str(output), *options])
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("compiler produced no native binary")
        binaries.append(
            {
                "path": output.relative_to(directory).as_posix(),
                "sha256": digest(output),
                "entrypoints": list(entrypoints),
            }
        )
    (directory / "bridge").mkdir(exist_ok=False)
    bridge = directory / "bridge" / f"{prepared['namespace']}.so"
    includes = [
        torch_root / "include",
        torch_root / "include/torch/csrc/api/include",
        npu / "include",
        cann / "include",
        Path(sysconfig.get_path("include")),
    ]
    libraries = [torch_root / "lib", npu / "lib", cann / "lib64"]
    run_compiler(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={prepared['namespace']}",
            f"-D_GLIBCXX_USE_CXX11_ABI={toolchain['abi']}",
            *[f"-I{path}" for path in includes],
            str(root / "tools/glm_perf/reconstruction_bridge.cpp"),
            *[f"-L{path}" for path in libraries],
            *[f"-Wl,-rpath,{path}" for path in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(bridge),
        ],
    )
    if not bridge.is_file() or not bridge.stat().st_size:
        raise RuntimeError("compiler produced no versioned bridge")
    bridges = [{"path": bridge.relative_to(directory).as_posix(), "sha256": digest(bridge)}]
    candidate = {
        key: prepared[key]
        for key in (
            "namespace",
            "assets",
            "base_reference_sha256",
            "configuration_sha256",
            "configuration",
            "contract_sha256",
        )
    }
    candidate.update(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "qwen_streaming_compiled_candidate",
            "binaries": sorted(binaries, key=lambda item: item["path"]),
            "bridges": bridges,
            "native_components": [name for name, _ in KERNELS],
            "prepared_manifest_sha256": prepared["manifest_sha256"],
            "hardware_validated": False,
            "admissible_without_gate_evidence": False,
        }
    )
    candidate["resources_sha256"] = canonical_sha256(resource_inventory(candidate))
    receipt = seal_manifest(
        {
            "kind": "host_only_compile_receipt",
            "namespace": prepared["namespace"],
            "prepared_manifest_sha256": prepared["manifest_sha256"],
            "resources_sha256": candidate["resources_sha256"],
            "target": "dav-2002",
            "compiler_options": options,
            "toolchain_versions": toolchain["versions"],
            "torch_abi": toolchain["abi"],
            "abi_evidence": frozen_abi,
            "toolchain_bytes": toolchain["byte_identities"],
            "compiler_sha256": digest(compiler),
            "filesystem_containment": containment,
            "command_attestations": command_attestations,
            "npu_opened": False,
            "library_loaded": False,
            "kernels_launched": False,
            "server_contacted": False,
        }
    )
    verify_bundle(directory)
    _write_new(directory / "build-receipt.json", receipt)
    candidate["build_receipt"] = {"path": "build-receipt.json", "sha256": digest(directory / "build-receipt.json")}
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    _write_new(directory / "candidate.json", candidate)
    return verify_bundle(directory, require_compiled=True)


def make_native_manifest(directory, validation_source):
    """Metadata only; admitted caller supplies its explicit deferred loader source."""
    if not isinstance(validation_source, str) or not validation_source.strip():
        raise ValueError("admitted caller must supply explicit deferred loader source")
    directory = Path(directory).resolve(strict=True)
    candidate = verify_bundle(directory, require_compiled=True)
    libraries = [
        {"path": str(_path(directory, entry["path"])), "sha256": entry["sha256"]} for entry in candidate["bridges"]
    ]
    assets = [
        {"path": str(_path(directory, entry["path"])), "sha256": entry["sha256"]}
        for entry in candidate["assets"] + candidate["binaries"]
    ]
    return {
        "name": candidate["namespace"],
        "libraries": libraries,
        "assets": assets,
        "operators": [f"{candidate['namespace']}::launch"],
        "validation_source": validation_source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--version", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--torch-abi", type=int, choices=(0, 1))
    parser.add_argument("--abi-evidence", type=Path)
    args = parser.parse_args()
    if args.source_root:
        prepare_bundle(
            args.bundle, args.source_root, args.version, args.reference, json.loads(args.configuration.read_text())
        )
    if args.compile:
        if args.cann_root is None:
            parser.error("--compile requires --cann-root")
        value = build_bundle(args.bundle, args.cann_root, args.torch_abi, args.abi_evidence)
    else:
        value = verify_bundle(args.bundle)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
