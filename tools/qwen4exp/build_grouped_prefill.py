# SPDX-License-Identifier: Apache-2.0
"""Freeze and build prefill-only scheduling and local-operand experiments."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tools.qwen4exp.build_native_prefill import build, digest


def build_grouped(directory, cann, root, version):
    directory, cann, root = directory.resolve(), cann.resolve(strict=True), root.resolve(strict=True)
    provenance = build(directory, cann, root, version)
    kernel_root = root / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"
    for source in [
        root / "tools/qwen4exp/native_grouped_prefill.cpp",
        *[
            kernel_root / name
            for name in (
                "native_int4_schedule.h",
                "native_int4_route_groups.h",
                "qwen_w4_a8_int4_matmul_v310_tiling_data.h",
            )
        ],
    ]:
        frozen = directory / source.name
        shutil.copy2(source, frozen)
        provenance["sources"][source.name] = digest(frozen)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binary = directory / "native_grouped_prefill.bin"
    subprocess.run(
        [
            str(directory / "compile-prefill"),
            str(directory / "native_grouped_prefill.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"-I{directory}",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    provenance["binaries"]["native_grouped_prefill"] = {"path": str(binary), "sha256": digest(binary)}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build_grouped(args.build_dir, args.cann_root, args.source_root, args.version), indent=2))
