# SPDX-License-Identifier: Apache-2.0
"""Freeze and compile the fused QSA metadata experiment with the native bridge."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .build_reconstruction import build as build_bridge
from .build_reconstruction import sha256


def build(build_dir, cann_root, source_root, version):
    build_bridge(build_dir, cann_root, source_root, version=version)
    path = build_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    cann = cann_root.resolve(strict=True)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    source = Path(__file__).parent
    frozen = build_dir / "glm_qsa_metadata.cpp"
    shutil.copy2(source / frozen.name, frozen)
    package = build_dir / provenance["_build"]["helper_package"]
    for name in ("qsa_metadata.py", "qsa_metadata_probe.py", "qsa_metadata_binding.py"):
        shutil.copy2(source / name, package / name)
        provenance["_helpers"][name] = sha256(package / name)
    binary = build_dir / "glm_qsa_metadata.bin"
    subprocess.run(
        [
            str(build_dir / "compile-reconstruction"),
            str(frozen),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    provenance["_build"]["fused_qsa_metadata"] = True
    provenance[binary.name] = {"source_sha256": sha256(frozen), "binary_sha256": sha256(binary)}
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return str(build_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    print(build(args.build_dir, args.cann_root, args.source_root, args.version))


if __name__ == "__main__":
    main()
