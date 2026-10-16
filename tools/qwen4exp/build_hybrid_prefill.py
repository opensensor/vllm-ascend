# SPDX-License-Identifier: Apache-2.0
"""Freeze experimental hybrid resources in a new namespace without opening devices."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tools.qwen4exp.build_grouped_prefill import build_grouped
from tools.qwen4exp.build_native_prefill import digest


def build_hybrid(directory, cann, root, version):
    provenance = build_grouped(directory, cann, root, version)
    for name in ("native_hybrid_prefill.cpp", "native_int8_projection_probe.cpp", "native_channel_activation.cpp"):
        frozen = directory / name
        shutil.copy2(root / "tools/qwen4exp" / name, frozen)
        provenance["sources"][name] = digest(frozen)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binary = directory / "native_hybrid_prefill.bin"
    subprocess.run(
        [
            str(directory / "compile-prefill"),
            str(directory / "native_hybrid_prefill.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"-I{directory}",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    provenance["binaries"]["native_hybrid_prefill"] = {"path": str(binary), "sha256": digest(binary)}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build_hybrid(args.build_dir, args.cann_root, args.source_root, args.version), indent=2))
