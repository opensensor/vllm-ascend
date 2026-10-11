# SPDX-License-Identifier: Apache-2.0
"""Build an isolated integer arithmetic probe without opening an NPU device."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths


def build(directory, cann, root, projection=False):
    """Freeze sources, compile isolated kernels/bridge, and record their hashes.

    A fresh directory preserves earlier binaries used by captured graphs.
    Compilation never registers kernels in serving workers or opens a device.
    The manifest records build provenance; separate probes establish accuracy.
    """
    directory.mkdir(parents=True, exist_ok=False)
    sources = [
        root / "tools/glm_perf/compile_reconstruction.cpp",
        root / "tools/glm_perf/reconstruction_bridge.cpp",
        root / "tools/qwen4exp/native_int8_group_probe.cpp",
    ]
    if projection:
        sources.append(root / "tools/qwen4exp/native_int8_projection_probe.cpp")
        sources.append(root / "tools/qwen4exp/native_int8_activation_probe.cpp")
    for source in sources:
        shutil.copy2(source, directory / source.name)
    compiler = directory / "compile-probe"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(directory / sources[0].name),
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
    projection_binary = directory / "int8-projection.bin"
    activation_binary = directory / "int8-activation.bin"
    if projection:
        for source, output in [(sources[3], projection_binary), (sources[4], activation_binary)]:
            subprocess.run(
                [
                    str(compiler),
                    str(directory / source.name),
                    str(output),
                    "--npu-arch=dav-2002",
                    f"--sysroot={cann / 'tools/hcc/sysroot'}",
                    f"-isystem{target}",
                    f"-isystem{target / 'aarch64-target-linux-gnu'}",
                ],
                check=True,
            )
    binary = directory / "int8-group.bin"
    subprocess.run(
        [
            str(compiler),
            str(directory / sources[2].name),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("use the serving torch-npu environment")
    npu = Path(spec.origin).parent
    includes = [*include_paths(), str(npu / "include"), str(cann / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu / "lib"), str(cann / "lib64")]
    bridge = directory / "qwen_int8_group_probe_v1.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            "-DGLM_RECONSTRUCTION_NAMESPACE=qwen_int8_group_probe_v1",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{path}" for path in includes],
            str(directory / sources[1].name),
            *[f"-L{path}" for path in libraries],
            *[f"-Wl,-rpath,{path}" for path in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-o",
            str(bridge),
        ],
        check=True,
    )
    manifest = {
        "namespace": "qwen_int8_group_probe_v1",
        "binary": str(binary),
        "bridge": str(bridge),
        "projection_binary": str(projection_binary) if projection else None,
        "activation_binary": str(activation_binary) if projection else None,
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                binary,
                bridge,
                *([projection_binary, activation_binary] if projection else []),
                *[directory / source.name for source in sources],
            ]
        },
        "hardware_validated": False,
        "server_modified": False,
    }
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--projection", action="store_true", help="also build the experimental full-K projection")
    args = parser.parse_args()
    build(args.build_dir.resolve(), args.cann_root.resolve(), args.source_root.resolve(), args.projection)
