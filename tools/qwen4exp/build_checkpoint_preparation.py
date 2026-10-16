# SPDX-License-Identifier: Apache-2.0
"""Compile an isolated checkpoint pre-copy experiment without opening an NPU device."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

NAMESPACE = "qwen_checkpoint_preparation_v2"
SOURCES = (
    "tools/glm_perf/compile_reconstruction.cpp",
    "tools/qwen4exp/native_checkpoint_preparation.cpp",
    "tools/qwen4exp/checkpoint_preparation_integer.h",
    "tools/qwen4exp/native_owned_bridge.cpp",
    "tools/qwen4exp/shared_launch_arguments.h",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(directory, cann, root):
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
    sources = {}
    for relative in SOURCES:
        source = root / relative
        frozen = directory / source.name
        shutil.copy2(source, frozen)
        sources[frozen.name] = digest(frozen)
    compiler = directory / "compile-checkpoint-preparation"
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
    binary = directory / "native_checkpoint_preparation.bin"
    subprocess.run(
        [
            str(compiler),
            str(directory / "native_checkpoint_preparation.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"-I{directory}",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
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
            "-DQWEN_NATIVE_FLUSH_HOST_QUEUE=0",
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
        "entry": NAMESPACE,
        "bridge_source": "native_owned_bridge.cpp",
        "flush_host_queue": False,
        "eager_only": False,
        "sources": sources,
        "binaries": {"native_checkpoint_preparation": {"path": str(binary), "sha256": digest(binary)}},
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
    args = parser.parse_args()
    print(json.dumps(build(args.build_dir, args.cann_root, args.source_root), indent=2))
