# SPDX-License-Identifier: Apache-2.0
"""Compile paired standalone beta kernels without selecting an NPU."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .build_query_bf16_vector import compile_bridge, digest


def build(output, compiler, namespace, version, cann_root):
    if type(version) is not int or version <= 0 or namespace != f"glm_kda_beta_scale_v{version}":
        raise ValueError("KDA beta diagnostic requires a fresh version namespace")
    compiler, cann = (path.resolve(strict=True) for path in (compiler, cann_root))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    assets = {}
    for name in ("glm_kda_beta_scale_vector.cpp", "kda_beta_scale_vector_probe.py", "reconstruction_bridge.cpp"):
        shutil.copy2(root / name, output / name)
        assets[name] = digest(output / name)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    for name in ("scalar", "vector"):
        source = output / f"kda_beta_{name}.cpp"
        source.write_text(
            ("#define GLM_KDA_BETA_SCALAR_REFERENCE\n" if name == "scalar" else "")
            + (output / "glm_kda_beta_scale_vector.cpp").read_text()
        )
        binary = output / f"kda_beta_{name}.bin"
        subprocess.run(
            [
                str(compiler),
                str(source),
                str(binary),
                "--npu-arch=dav-2002",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
            ],
            check=True,
        )
        assets[source.name], assets[binary.name] = digest(source), digest(binary)
    bridge = output / f"glm_kda_beta_scale_bridge_v{version}.so"
    compile_bridge(bridge, namespace, cann)
    assets[bridge.name] = digest(bridge)
    provenance = dict(version=version, namespace=namespace, assets=assets, compile_only=True, full_kda_evaluated=False)
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
