# SPDX-License-Identifier: Apache-2.0
"""Compile isolated four/eight-block QSA gathers without opening an NPU device."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

NAMESPACE = "qwen_qsa_gather_staging_v1"
SOURCES = (
    "tools/glm_perf/compile_reconstruction.cpp",
    "tools/qwen4exp/native_qsa_gather_staging.cpp",
    "tools/qwen4exp/native_owned_bridge.cpp",
    "tools/qwen4exp/shared_launch_arguments.h",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


QUALIFIED_HEADERS = {
    "qsa_gather_value_nz_v310.h": "dbceeef186d29656390a644061f781ecb682ad4ae5a9d83f0c4969db169671a1",
    "qsa_gather_value_nz_v310_tiling_data.h": "5193b22a2ca386b1fff93f807472090a2f8e202b1a11d32477bf0dd0ac9d3681",
}
STAGING_DECLARATION = "constexpr int64_t TOKEN_BLOCKS_PER_TILE = 4;"


def freeze_headers(runtime, directory):
    headers = Path(runtime) / "csrc/attention/qsa_gather_value_nz_v310/op_kernel"
    for name, expected in QUALIFIED_HEADERS.items():
        if digest(headers / name) != expected:
            raise ValueError(f"qualified QSA gather header changed: {name}")
    original = (headers / "qsa_gather_value_nz_v310.h").read_text()
    if original.count(STAGING_DECLARATION) != 1:
        raise ValueError("QSA staging declaration must occur exactly once")
    sources = {}
    for name in QUALIFIED_HEADERS:
        shutil.copy2(headers / name, directory / name)
        sources[name] = digest(directory / name)
    for blocks in (4, 8):
        frozen = directory / f"qsa_gather_staging_{blocks}.h"
        frozen.write_text(original.replace(STAGING_DECLARATION, f"constexpr int64_t TOKEN_BLOCKS_PER_TILE = {blocks};"))
        sources[frozen.name] = digest(frozen)
    return sources


def build(directory, cann, root, runtime):
    # CPU compilation needs torch headers, never torch_npu device initialization.
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise RuntimeError("CPU-only compilation requires upstream TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    root, cann = (Path(p).resolve(strict=True) for p in (root, cann))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("matching torch-npu package headers required")
    sources = freeze_headers(runtime, directory)
    for relative in SOURCES:
        source = root / relative
        frozen = directory / source.name
        shutil.copy2(source, frozen)
        sources[frozen.name] = digest(frozen)
    compiler = directory / "compile-qsa-gather-staging"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(directory / "compile_reconstruction.cpp"),
            f"-I{cann / 'include'}",
            f"-L{cann / 'lib64'}",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-o",
            str(compiler),
        ],
        check=True,
    )
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binaries = {}
    for blocks in (4, 8):
        name = f"native_qsa_gather_staging_{blocks}"
        binary = directory / f"{name}.bin"
        subprocess.run(
            [
                str(compiler),
                str(directory / "native_qsa_gather_staging.cpp"),
                str(binary),
                "--npu-arch=dav-2002",
                f"-DQSA_STAGING_BLOCKS={blocks}",
                f"-I{directory}",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
            ],
            check=True,
        )
        binaries[name] = {
            "path": str(binary),
            "sha256": digest(binary),
            "entry": f"qwen_qsa_gather_staging_{blocks}_v1",
        }
    npu = Path(spec.origin).parent
    includes = [*include_paths(), str(npu / "include"), str(cann / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
    bridge = directory / f"{NAMESPACE}.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={NAMESPACE}",
            "-DQWEN_NATIVE_FLUSH_HOST_QUEUE=1",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{path}" for path in includes],
            str(directory / "native_owned_bridge.cpp"),
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
        check=True,
    )
    provenance = {
        "namespace": NAMESPACE,
        "native_task_type": "KERNEL_TYPE_AICORE",
        "bridge_source": "native_owned_bridge.cpp",
        "flush_host_queue": True,
        "eager_only": True,
        "sources": sources,
        "binaries": binaries,
        "qualified_headers": dict(QUALIFIED_HEADERS),
        "bridge": {"path": str(bridge), "sha256": digest(bridge)},
        "hardware_validated": False,
        "server_modified": False,
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--qualified-runtime", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.build_dir, args.cann_root, args.source_root, args.qualified_runtime), indent=2))
