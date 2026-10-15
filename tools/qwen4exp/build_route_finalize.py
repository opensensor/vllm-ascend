# SPDX-License-Identifier: Apache-2.0
"""Build two isolated W4 prefill experiments without opening an NPU device."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

QUALIFIED_HEADERS = {
    "native_int4_schedule.h": "cebf66752cfe9a9606a584dab07a25dbf8a4f8a8d4057ad0a372dfb65c7a7f70",
    "qwen_w4_a8_int4_matmul_v310_tiling_data.h": "75355de444258e0edbc53dc917d3338d646a827ec8aba441782654e383686e53",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_headers(runtime, destination):
    """Refuse the newer working tree's changed schedule or descriptor ABI."""
    kernel_root = runtime / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"
    for name, expected in QUALIFIED_HEADERS.items():
        if digest(kernel_root / name) != expected:
            raise ValueError(f"qualified prefill header changed: {name}")
    for name in QUALIFIED_HEADERS:
        shutil.copy2(kernel_root / name, destination / name)
    return dict(QUALIFIED_HEADERS)


def build(directory, cann, root, runtime, version, bridge_source=None, *, flush_host_queue=False):
    # Compilation needs torch headers, not torch_npu import/device state.
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise RuntimeError("CPU-only compilation requires upstream TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    if type(version) is not int or version <= 0:
        raise ValueError("version must be a positive integer")
    if type(flush_host_queue) is not bool:
        raise ValueError("flush_host_queue must be a boolean")
    if flush_host_queue and (
        version != 3 or bridge_source is None or Path(bridge_source).name != "native_owned_bridge.cpp"
    ):
        raise ValueError("host queue flushing requires version 3 and the explicit shared-owner bridge")
    root, runtime, cann = (p.resolve(strict=True) for p in (root, runtime, cann))
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = freeze_headers(runtime, directory)
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("matching torch-npu package headers required")
    namespace = f"qwen_route_finalize_v{version}"
    bridge_source = (
        root / "tools/glm_perf/reconstruction_bridge.cpp"
        if bridge_source is None
        else Path(bridge_source).resolve(strict=True)
    )
    if bridge_source.name == "native_owned_bridge.cpp" and version < 2:
        raise ValueError("shared-owner bridge requires a fresh namespace version >=2")
    paths = (
        "tools/glm_perf/compile_reconstruction.cpp",
        "tools/qwen4exp/native_route_finalize.cpp",
        "tools/qwen4exp/native_prefill_schedule_probe.cpp",
    )
    for relative in paths:
        source = root / relative
        frozen = directory / source.name
        shutil.copy2(source, frozen)
        sources[frozen.name] = digest(frozen)
    shutil.copy2(bridge_source, directory / bridge_source.name)
    sources[bridge_source.name] = digest(directory / bridge_source.name)
    if bridge_source.name == "native_owned_bridge.cpp":
        helper = bridge_source.parent / "shared_launch_arguments.h"
        shutil.copy2(helper, directory / helper.name)
        sources[helper.name] = digest(directory / helper.name)
    compiler = directory / "compile-route-finalize"
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
    options = [
        "--npu-arch=dav-2002",
        f"-I{directory}",
        f"--sysroot={cann / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
    ]
    binaries = {}
    for name in ("native_route_finalize", "native_prefill_schedule_probe"):
        binary = directory / f"{name}.bin"
        subprocess.run([str(compiler), str(directory / f"{name}.cpp"), str(binary), *options], check=True)
        binaries[name] = {"path": str(binary), "sha256": digest(binary)}
    npu = Path(spec.origin).parent
    includes = [*include_paths(), str(npu / "include"), str(cann / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
    bridge = directory / f"{namespace}.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={namespace}",
            *(["-DQWEN_NATIVE_FLUSH_HOST_QUEUE=1"] if flush_host_queue else []),
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{path}" for path in includes],
            str(directory / bridge_source.name),
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
        "namespace": namespace,
        "bridge_source": bridge_source.name,
        "flush_host_queue": flush_host_queue,
        "sources": sources,
        "binaries": binaries,
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
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--bridge-source", type=Path)
    parser.add_argument("--flush-host-queue", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.build_dir,
                args.cann_root,
                args.source_root,
                args.qualified_runtime,
                args.version,
                args.bridge_source,
                flush_host_queue=args.flush_host_queue,
            ),
            indent=2,
        )
    )
