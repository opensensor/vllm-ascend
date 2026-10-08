# SPDX-License-Identifier: Apache-2.0
"""Compile an append-only BF16 conversion bundle without initializing an NPU."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .build_query_bf16_vector import compile_bridge, digest


def build(output, compiler, namespace, version, cann_root):
    if type(version) is not int or version <= 0 or namespace != f"glm_bf16_vector_v{version}":
        raise ValueError("fresh tiled BF16 conversion requires a unique version namespace")
    compiler, cann = (path.resolve(strict=True) for path in (compiler, cann_root))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    assets = {}
    for name in ("glm_bf16_vector.cpp", "bf16_vector.py", "bf16_vector_probe.py", "reconstruction_bridge.cpp"):
        shutil.copy2(source / name, output / name)
        assets[name] = digest(output / name)
    binary = output / "glm_bf16_vector.bin"
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    subprocess.run(
        [
            str(compiler),
            str(output / "glm_bf16_vector.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    assets[binary.name] = digest(binary)
    bridge = output / f"glm_bf16_vector_bridge_v{version}.so"
    compile_bridge(bridge, namespace, cann)
    assets[bridge.name] = digest(bridge)
    provenance = dict(version=version, namespace=namespace, assets=assets, compile_only=True, hardware_gates="not_run")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "compiler", "cann-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    print(build(args.build_dir, args.compiler, args.namespace, args.version, args.cann_root))


if __name__ == "__main__":
    main()
