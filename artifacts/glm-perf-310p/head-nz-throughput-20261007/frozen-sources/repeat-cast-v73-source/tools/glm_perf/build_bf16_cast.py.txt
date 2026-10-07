# SPDX-License-Identifier: Apache-2.0
"""Build an append-only AI-Core BF16 converter using an existing ACLRTC compiler."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

HERE = Path(__file__).resolve().parent


def build(output, compiler, version, cann_root):
    if type(version) is not int or version <= 0:
        raise ValueError("BF16 kernel version must be positive")
    compiler, cann = compiler.resolve(strict=True), cann_root.resolve(strict=True)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source, name in (
        (HERE / "glm_bf16_cast.cpp", "glm_bf16_cast.cpp"),
        (HERE / "bf16_cast.py", "bf16_cast.py"),
        (HERE / "resident_candidates/indexer_aicore_casts.py", "candidate.py"),
    ):
        shutil.copy2(source, output / name)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    subprocess.run(
        [
            str(compiler),
            str(output / "glm_bf16_cast.cpp"),
            str(output / "glm_bf16_cast.bin"),
            "--npu-arch=dav-2002",
            f"--sysroot={cann}/tools/hcc/sysroot",
            f"-isystem{target}",
            f"-isystem{target}/aarch64-target-linux-gnu",
        ],
        check=True,
    )
    npu = Path(importlib.util.find_spec("torch_npu").origin).parent
    includes = [*include_paths(), npu / "include", cann / "include", sysconfig.get_path("include")]
    libraries = [*library_paths(), npu / "lib", cann / "lib64"]
    namespace, bridge = f"glm_bf16_v{version}", f"glm_bf16_bridge_v{version}.so"
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
            str(HERE / "reconstruction_bridge.cpp"),
            *[f"-L{path}" for path in libraries],
            *[f"-Wl,-rpath,{path}" for path in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(output / bridge),
        ],
        check=True,
    )
    (output / "options.json").write_text(
        json.dumps({"name": f"bf16_cast_v{version}", "namespace": namespace, "bridge": bridge}, indent=2) + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    args = parser.parse_args()
    print(build(args.build_dir, args.compiler, args.version, args.cann_root))


if __name__ == "__main__":
    main()
