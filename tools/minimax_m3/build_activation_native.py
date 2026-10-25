# SPDX-License-Identifier: Apache-2.0
"""Build an append-only, sealed clipped SwiGLU kernel without opening a device."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tools.glm_perf.build_query_bf16_vector import compile_bridge
from tools.glm_perf.resident_native import file_digest


def build(output, compiler, namespace, cann):
    if not namespace.isidentifier() or not namespace.startswith("minimax_activation_"):
        raise ValueError("use a unique minimax_activation_ namespace")
    output, compiler, cann = output.resolve(), compiler.resolve(strict=True), cann.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for name in ("native_activation.cpp", "activation_native.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    shutil.copy2(root / "glm_perf/channel_scale_bridge.cpp", output / "reconstruction_bridge.cpp")
    shutil.copy2(root.parent / "csrc/common/include/cann_compat.h", output / "cann_compat.h")
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binary = output / "activation.bin"
    subprocess.run(
        [
            str(compiler),
            str(output / "native_activation.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
            f"-include{output / 'cann_compat.h'}",
        ],
        check=True,
    )
    compile_bridge(output / "bridge.so", namespace, cann)
    helper = output / "activation_native.py"
    validation = (
        "import types\n"
        f"_module = types.ModuleType({namespace!r})\n"
        f"exec(compile(open({str(helper)!r}).read(), {str(helper)!r}, 'exec'), _module.__dict__)\n"
        f"def prepare():\n    return _module.ClippedSwiGLUKernel({namespace!r}, {str(binary)!r})\n"
        "def validate(resource):\n    return _module.validate_kernel(resource)\n"
    )

    def record(path):
        return {"path": str(path), "sha256": file_digest(path)}

    manifest = {
        "name": namespace,
        "libraries": [record(output / "bridge.so")],
        "assets": [
            record(output / name)
            for name in (
                "activation.bin",
                "native_activation.cpp",
                "activation_native.py",
                "reconstruction_bridge.cpp",
                "cann_compat.h",
            )
        ],
        "operators": [namespace + "::launch"],
        "validation_source": validation,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    args = parser.parse_args()
    print(build(args.output, args.compiler, args.namespace, args.cann))


if __name__ == "__main__":
    main()
