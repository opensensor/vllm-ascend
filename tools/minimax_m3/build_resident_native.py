# SPDX-License-Identifier: Apache-2.0
"""Compile and seal a versioned direct CANN W4 kernel without opening a device."""

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from tools.glm_perf.build_query_bf16_vector import compile_bridge
from tools.glm_perf.resident_native import file_digest


def build(output, source_root, namespace, cann):
    if not namespace.isidentifier() or not namespace.startswith("minimax_w4_"):
        raise ValueError("use a unique minimax_w4_ namespace for each build")
    output, source_root, cann = output.resolve(), source_root.resolve(strict=True), cann.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    for directory in (
        "gmm/w2_blocked_dequant_matmul_v310/op_kernel",
        "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel",
        "third_party/catlass/include",
        "moe/common/kernel_utils",
        "common/include",
    ):
        shutil.copytree(source_root / directory, sources / directory)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "glm_perf/compile_reconstruction.cpp", sources / "compiler.cpp")
    shutil.copy2(root / "glm_perf/channel_scale_bridge.cpp", output / "reconstruction_bridge.cpp")
    compiler = output / "compile-kernel"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(sources / "compiler.cpp"),
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
    grouped = sources / "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel"
    blocked = sources / "gmm/w2_blocked_dequant_matmul_v310/op_kernel"
    kernel_source = (grouped / "w2_grouped_blocked_dequant_matmul_v310.cpp").read_text()
    entry = namespace + "_kernel"
    # OPP normally initializes a hidden system-workspace pointer. A direct
    # binary owns its workspace explicitly and must never depend on that state.
    old = "GM_ADDR user = AscendC::GetUserWorkspace(workspace);"
    if kernel_source.count(old) != 1:
        raise ValueError("grouped kernel workspace contract changed")
    kernel_source = kernel_source.replace(old, "GM_ADDR user = workspace;")
    kernel_source = kernel_source.replace("KERNEL_TYPE_MIX_AIC", "KERNEL_TYPE_AICORE")
    kernel_source = kernel_source.replace("void w2_grouped_blocked_dequant_matmul_v310(", f"void {entry}(")
    source = sources / "grouped_direct.cpp"
    source.write_text(kernel_source)
    binary = output / "grouped_w4.bin"
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    subprocess.run(
        [
            str(compiler),
            str(source),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
            f"-I{grouped}",
            f"-I{blocked}",
            f"-I{sources / 'third_party/catlass/include'}",
            f"-I{sources / 'moe/common'}",
            f"-include{sources / 'common/include/cann_compat.h'}",
            "-Wno-ignored-attributes",
        ],
        check=True,
    )
    compile_bridge(output / "bridge.so", namespace, cann)
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        archive.add(sources, arcname="sources")
    helper = output / "resident_native.py"
    shutil.copy2(Path(__file__).with_name("resident_native.py"), helper)
    validation = (
        "import types\n"
        f"_module = types.ModuleType({namespace!r})\n"
        f"exec(compile(open({str(helper)!r}).read(), {str(helper)!r}, 'exec'), _module.__dict__)\n"
        f"def prepare():\n    return _module.GroupedW4Kernel({namespace!r}, {str(binary)!r}, {entry!r})\n"
        "def validate(resource):\n    return _module.validate_kernel(resource)\n"
    )

    def record(path):
        return {"path": str(path), "sha256": file_digest(path)}

    manifest = {
        "name": namespace,
        "libraries": [record(output / "bridge.so")],
        "assets": [
            record(output / name)
            for name in ("grouped_w4.bin", "sources.tar.gz", "resident_native.py", "reconstruction_bridge.cpp")
        ],
        "operators": [namespace + "::launch"],
        "validation_source": validation,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2] / "csrc")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    args = parser.parse_args()
    print(build(args.output, args.source_root, args.namespace, args.cann))


if __name__ == "__main__":
    main()
