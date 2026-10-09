# SPDX-License-Identifier: Apache-2.0
"""Build an append-only FP32-output mHC projection, without opening a device."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .build_reconstruction import build as build_bridge
from .build_reconstruction import sha256


def build(directory, cann_root, source_root, version, *, tile_k=256):
    if type(tile_k) is not int or tile_k not in (256, 512, 1024):
        raise ValueError("projection K tile must be 256, 512 or 1024")
    build_bridge(directory, cann_root, source_root, version=version)
    cann = cann_root.resolve(strict=True)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    source = Path(__file__).parent / "mhc_projection_kernel.cpp"
    frozen = directory / source.name
    shutil.copy2(source, frozen)
    binary = directory / "glm_mhc_projection.bin"
    command = [
        str(directory / "compile-reconstruction"),
        str(frozen),
        str(binary),
        "--npu-arch=dav-2002",
        f"--sysroot={cann / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
        f"-DGLM_MHC_PROJECTION_TILE_K={tile_k}",
    ]
    subprocess.run(command, check=True)
    path = directory / "provenance.json"
    provenance = json.loads(path.read_text())
    package = directory / provenance["_build"]["helper_package"]
    helper = package / "mhc_projection.py"
    shutil.copy2(source.with_name(helper.name), helper)
    provenance["_helpers"][helper.name] = sha256(helper)
    provenance["_build"]["mhc_fp32_projection"] = True
    provenance["_build"]["mhc_projection_tile_k"] = tile_k
    provenance[binary.name] = dict(source_sha256=sha256(frozen), binary_sha256=sha256(binary), command=command)
    path.write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "cann-root", "source-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--tile-k", type=int, choices=(256, 512, 1024), default=256)
    args = parser.parse_args()
    build(args.build_dir, args.cann_root, args.source_root, args.version, tile_k=args.tile_k)


if __name__ == "__main__":
    main()
