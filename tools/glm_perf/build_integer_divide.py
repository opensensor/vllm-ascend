# SPDX-License-Identifier: Apache-2.0
"""Append-only integer metadata kernel with a frozen qualified bridge."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .build_reconstruction import build as build_bridge
from .build_reconstruction import sha256
from .integer_divide import INTEGER_DIVIDE_ENTRY


def build(directory, cann_root, source_root, version):
    build_bridge(directory, cann_root, source_root, version=version)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    cann = cann_root.resolve(strict=True)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    source = Path(__file__).parent
    frozen = directory / "glm_integer_divide.cpp"
    shutil.copy2(source / frozen.name, frozen)
    binary = directory / "glm_integer_divide.bin"
    subprocess.run(
        [
            str(directory / "compile-reconstruction"),
            str(frozen),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    package = directory / provenance["_build"]["helper_package"]
    for name in ("integer_divide.py", "integer_divide_probe.py"):
        shutil.copy2(source / name, package / name)
        provenance["_helpers"][name] = sha256(package / name)
    provenance["_build"]["integer_metadata_divide"] = True
    provenance["_build"]["integer_divide_entry"] = INTEGER_DIVIDE_ENTRY
    provenance[binary.name] = {"source_sha256": sha256(frozen), "binary_sha256": sha256(binary)}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("build-dir", "cann-root", "source-root"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    build(args.build_dir, args.cann_root, args.source_root, args.version)


if __name__ == "__main__":
    main()
