# SPDX-License-Identifier: Apache-2.0
"""CPU-only ACLRTC compilation; no device selection, kernel load or inference."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_bridge(output, namespace, cann):
    """Compile a unique namespace without importing torch-npu or loading it."""
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("bridge compilation requires the serving torch-npu headers")
    npu = Path(spec.origin).parent
    includes = [*include_paths(), npu / "include", cann / "include", sysconfig.get_path("include")]
    libraries = [*library_paths(), npu / "lib", cann / "lib64"]
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
            str(output.parent / "reconstruction_bridge.cpp"),
            *[f"-L{path}" for path in libraries],
            *[f"-Wl,-rpath,{path}" for path in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(output),
        ],
        check=True,
    )


def build(output, compiler, bridge, namespace, version, cann_root):
    if type(version) is not int or version <= 0 or not namespace.isidentifier():
        raise ValueError("positive build version and explicit bridge namespace required")
    compiler, cann = (path.resolve(strict=True) for path in (compiler, cann_root))
    if bridge is not None:
        bridge = bridge.resolve(strict=True)
    elif namespace != f"glm_query_vector_v{version}":
        raise ValueError("a fresh bridge requires the unique query-vector version namespace")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    assets = {}
    for name in ("glm_query_bf16_vector.cpp", "query_bf16_vector.py", "query_bf16_vector_probe.py"):
        shutil.copy2(root / name, output / name)
        assets[name] = digest(output / name)
    package_name = f"glm_query_vector_v{version}_helpers"
    package = output / package_name
    package.mkdir()
    (package / "__init__.py").write_text("# SPDX-License-Identifier: Apache-2.0\n")
    assets[package_name + "/__init__.py"] = digest(package / "__init__.py")
    for source, target_name in (
        ("instance_bindings.py", "instance_bindings.py"),
        ("query_cast_binding.py", "query_cast_binding.py"),
        ("resident_rpc_guard.py", "resident_rpc_guard.py"),
        ("resident_candidates/query_vector.py", "query_vector.py"),
    ):
        shutil.copy2(root / source, package / target_name)
        if target_name == "query_vector.py":
            # This immutable package is flat; keep all dependencies private to
            # the build instead of relying on an older installed tools package.
            path = package / target_name
            path.write_text(path.read_text().replace("from ..", "from ."))
        assets[package_name + "/" + target_name] = digest(package / target_name)
    binary = output / "glm_query_bf16_vector.bin"
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    subprocess.run(
        [
            str(compiler),
            str(output / "glm_query_bf16_vector.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    assets[binary.name] = digest(binary)
    reused_bridge = bridge is not None
    if bridge is None:
        shutil.copy2(root / "reconstruction_bridge.cpp", output / "reconstruction_bridge.cpp")
        assets["reconstruction_bridge.cpp"] = digest(output / "reconstruction_bridge.cpp")
        bridge = output / f"glm_query_vector_bridge_v{version}.so"
        compile_bridge(bridge, namespace, cann)
    provenance = dict(
        version=version,
        helper_package=package_name,
        namespace=namespace,
        compile_only=True,
        hardware_gates="not_run",
        reused_bridge=reused_bridge,
        bridge={"path": str(bridge), "sha256": digest(bridge)},
        assets=assets,
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, help="optional existing bridge for standalone probes only")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.build_dir, args.compiler, args.bridge, args.namespace, args.version, args.cann_root))


if __name__ == "__main__":
    main()
