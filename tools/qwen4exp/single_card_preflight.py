# SPDX-License-Identifier: Apache-2.0
"""Check a single-card bundle and host environment without initializing NPUs.

This checks critical SDK files and exported symbols, not the entire CANN
distribution. A passing report is not hardware or inference qualification.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from tools.qwen4exp.verify_single_card_bundle import verify_bundle

CORE_PACKAGES = ("torch", "torch-npu", "transformers", "vllm")
HCCL_SYMBOLS = ("HcclGetRootInfo", "HcclBroadcast", "HcclAllReduce")
MAX_LIBRARY_DEPENDENCIES = 64
BUILD_INFO = "# Prepared for the frozen Atlas 300I Duo runtime.\n__device_type__ = '_310P'\n"
ENVIRONMENT_KEYS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ATB_HOME_PATH",
    "SOC_VERSION",
    "VLLM_USE_V2_MODEL_RUNNER",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "HCCL_SOCKET_IFNAME",
    "HCCL_IF_IP",
    "HCCL_IF_BASE_PORT",
    "ASCEND_CUSTOM_OPP_PATH",
)


def normalize_name(name):
    return name.lower().replace("_", "-").replace(".", "-")


def build_device_type(path):
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__device_type__" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def prepare_build_info(bundle):
    """Add omitted generated metadata after validating the original snapshot."""
    count = verify_bundle(bundle)
    path = bundle / "runtime/vllm_ascend/_build_info.py"
    if path.exists():
        if build_device_type(path) != "_310P":
            raise ValueError("existing bundle build metadata does not select _310P; refusing to overwrite")
        return False
    with path.open("x") as output:
        output.write(BUILD_INFO)
    receipt = {
        "verified_original_files": count,
        "original_manifest_sha256": hashlib.sha256((bundle / "files.sha256.json").read_bytes()).hexdigest(),
        "added_file": "runtime/vllm_ascend/_build_info.py",
        "added_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (bundle / "build-info-preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return True


def compare_packages(expected, installed, critical):
    expected = {normalize_name(name): version for name, version in expected.items()}
    installed = {normalize_name(name): version for name, version in installed.items()}
    errors, warnings = [], []
    for name, version in expected.items():
        actual = installed.get(name)
        if actual != version:
            message = f"{name}: expected {version}; installed {actual or 'MISSING'}"
            (errors if name in critical else warnings).append(message)
    extras = sorted(set(installed) - set(expected))
    if extras:
        warnings.append("additional packages: " + ", ".join(extras))
    if "triton" in installed or "triton-ascend" in installed:
        errors.append("the frozen single-card profile was qualified without Triton; use a separate environment")
    return errors, warnings


def library_path(name, environment, fallbacks):
    paths = environment.get("LD_LIBRARY_PATH", "").split(":")
    for directory in [*paths, *map(str, fallbacks)]:
        if directory and (Path(directory) / name).is_file():
            return (Path(directory) / name).resolve()
    return None


def exported_symbols(path):
    # Inspect ELF without dlopen, importing torch or opening an NPU context.
    result = subprocess.run(
        ["nm", "-D", "--defined-only", str(path)], capture_output=True, text=True, timeout=20, check=True
    )
    return {line.split()[-1].split("@")[0] for line in result.stdout.splitlines() if line.split()}


def needed_libraries(path):
    result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=20, check=True)
    return [line.split("[", 1)[1].split("]", 1)[0] for line in result.stdout.splitlines() if "(NEEDED)" in line]


def hccl_symbol_graph(path, environment, sdk, symbol_reader, dependency_reader):
    """Inspect dependency exports too: CANN can place RootInfo in a companion."""
    pending, inspected, symbols = [path], [], set()
    while pending and not set(HCCL_SYMBOLS) <= symbols:
        current = pending.pop()
        if current in inspected:
            continue
        if len(inspected) >= MAX_LIBRARY_DEPENDENCIES:
            raise ValueError("HCCL dependency inspection exceeded the bounded library limit")
        inspected.append(current)
        symbols.update(symbol_reader(current))
        if set(HCCL_SYMBOLS) <= symbols:
            break
        for name in dependency_reader(current):
            dependency = library_path(
                name,
                environment,
                (current.parent, sdk / "lib64", Path("/lib/x86_64-linux-gnu"), Path("/usr/lib/x86_64-linux-gnu")),
            )
            if dependency is not None:
                pending.append(dependency)
    return symbols, inspected


def sdk_checks(environment, symbol_reader=exported_symbols, dependency_reader=needed_libraries):
    sdk = Path(environment.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/ascend-toolkit/latest")
    errors, warnings = [], []
    for relative in ("lib64/libascendcl.so", "opp", "devlib", "python/site-packages/tbe"):
        if not (sdk / relative).exists():
            errors.append(f"missing critical CANN component: {sdk / relative}")
    compiler_paths = (sdk / "compiler/bin/asc_opc", sdk / "compiler/ccec_compiler/bin/ccec")
    if not any(path.is_file() for path in compiler_paths):
        errors.append("missing CANN compiler entry point: asc_opc or ccec")
    topi_paths = (sdk / "python/site-packages/topi", sdk / "python/site-packages/tbe/tvm/topi")
    if not any(path.exists() for path in topi_paths):
        errors.append("missing CANN topi package in the expected SDK layout")
    atb = Path(environment.get("ATB_HOME_PATH") or "/usr/local/Ascend/nnal/atb/latest/atb")
    resolved = {}
    hccl_files = []
    for name, fallbacks in (
        ("libhccl.so", (sdk / "lib64",)),
        ("libatb.so", (atb / "lib", atb / "lib64")),
        ("libacl_dvpp.so", (sdk / "lib64",)),
    ):
        path = library_path(name, environment, fallbacks)
        resolved[name] = str(path) if path else None
        if path is None:
            errors.append(f"{name} is missing from the selected library search paths")
        elif name == "libhccl.so":
            try:
                symbols, hccl_files = hccl_symbol_graph(path, environment, sdk, symbol_reader, dependency_reader)
                missing = set(HCCL_SYMBOLS) - symbols
                if missing:
                    errors.append(f"{path}: missing exports {', '.join(sorted(missing))}")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(f"cannot verify HCCL exports with nm/readelf: {error}")
    if any("libhccl" in item for item in environment.get("LD_PRELOAD", "").replace(":", " ").split()):
        errors.append("remove HCCL from LD_PRELOAD; the reported workaround uses LD_LIBRARY_PATH instead")
    if environment.get("HCCL_SOCKET_IFNAME") == "lo" and environment.get("HCCL_IF_IP") != "127.0.0.1":
        warnings.append("single-host loopback recipe uses HCCL_IF_IP=127.0.0.1; verify the configured address")
    return {
        "root": str(sdk.resolve()),
        "libraries": resolved,
        "hccl_symbol_files": list(map(str, hccl_files)),
        "errors": errors,
        "warnings": warnings,
    }


def preflight(bundle, *, environment, installed, machine=None, python_version=None, symbol_reader=exported_symbols):
    expected_env = json.loads((bundle / "environment.json").read_text())
    inventory = json.loads((bundle / "python-packages.json").read_text())
    observed = {item["name"]: item["version"] for item in inventory}
    errors, warnings = compare_packages(observed, installed, CORE_PACKAGES)
    expected_python = expected_env["python"]
    actual_python = python_version or platform.python_version()
    if actual_python != expected_python:
        errors.append(f"Python: expected {expected_python}; running {actual_python}")
    if (machine or platform.machine()) != "x86_64":
        errors.append("the compiled bundle targets x86_64")
    path = bundle / "runtime/vllm_ascend/_build_info.py"
    try:
        if build_device_type(path) != "_310P":
            errors.append("missing or wrong _build_info.py; prepare the pristine bundle before launch")
    except (SyntaxError, ValueError) as error:
        errors.append(f"invalid build metadata: {error}")
    if environment.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        errors.append("FULL_DECODE_ONLY requires explicit VLLM_USE_V2_MODEL_RUNNER=1 for this profile")
    sdk = sdk_checks(environment, symbol_reader)
    errors.extend(sdk["errors"])
    warnings.extend(sdk["warnings"])
    return {
        "status": "blocked" if errors else "passed",
        "scope": "offline environment and critical SDK checks; not hardware qualification",
        "errors": errors,
        "warnings": warnings,
        "sdk": sdk,
        "environment": {key: environment.get(key) for key in ENVIRONMENT_KEYS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--prepare-build-info", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.prepare_build_info:
            prepare_build_info(args.bundle)
        installed = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
        report = preflight(args.bundle, environment=os.environ, installed=installed)
    except (OSError, ValueError, SyntaxError) as error:
        parser.error(str(error))
    serialized = json.dumps(report, indent=2) + "\n"
    print(serialized, end="")
    if args.output:
        args.output.write_text(serialized)
    if report["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
