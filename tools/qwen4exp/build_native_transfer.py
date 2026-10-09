# SPDX-License-Identifier: Apache-2.0
"""Compile append-only prefill resources offline; never opens an NPU device."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(directory: Path, cann: Path, root: Path, version: int):
    # Torch headers are needed for the bridge, without loading torch_npu or
    # opening a device. ACLRTC compilation is host work only.
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    if type(version) is not int or version <= 0:
        raise ValueError("version must be a positive integer")
    cann, root = cann.resolve(strict=True), root.resolve(strict=True)
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("compile using the matching serving torch-npu installation")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    namespace = f"qwen_transfer_v{version}"
    compiler_source = root / "tools/glm_perf/compile_reconstruction.cpp"
    bridge_source = root / "tools/glm_perf/reconstruction_bridge.cpp"
    sources = {}
    for path in (
        compiler_source,
        bridge_source,
        root / "tools/qwen4exp/native_state_layout.cpp",
        root / "tools/qwen4exp/native_cached_metadata.cpp",
        *sorted((root / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel").glob("*.h")),
    ):
        frozen = directory / path.name
        shutil.copy2(path, frozen)
        sources[path.name] = digest(frozen)
    compiler = directory / "compile-prefill"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(directory / compiler_source.name),
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
    for name in ("native_state_layout", "native_cached_metadata"):
        output = directory / f"{name}.bin"
        subprocess.run([str(compiler), str(directory / f"{name}.cpp"), str(output), *options], check=True)
        binaries[name] = {"path": str(output), "sha256": digest(output)}
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
        "sources": sources,
        "binaries": binaries,
        "bridge": {"path": str(bridge), "sha256": digest(bridge)},
        "hardware_validated": False,
        "server_modified": False,
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.build_dir, args.cann_root, args.source_root, args.version), indent=2))


if __name__ == "__main__":
    main()
