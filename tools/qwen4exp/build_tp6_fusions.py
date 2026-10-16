# SPDX-License-Identifier: Apache-2.0
"""Freeze and compile private TP6 fusions without loading a device or server."""

import argparse
import json
import subprocess
from pathlib import Path

from tools.qwen4exp.tp6_fusion_sources import digest, sources

KERNELS = ("native_tp6_gate_pack.cpp", "native_tp6_finalize.cpp")


def stage(root, destination):
    root, destination = Path(root), Path(destination)
    files = sources(root)
    files.update({name: (root / "tools/qwen4exp" / name).read_bytes() for name in KERNELS})
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    receipt = dict(
        schema_version=1,
        sources={n: digest(d) for n, d in files.items()},
        binaries={},
        hardware_validated=False,
        device_opened=False,
        server_modified=False,
        entries=dict(
            gate_pack="qwen_tp6_gate_pack_v1",
            ordered="qwen_tp6_route_finalize_v1",
            sequential="qwen_tp6_route_reduce_sequential_v1",
            pairwise="qwen_tp6_route_reduce_pairwise_v1",
        ),
    )
    (destination / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def compile_staged(destination, compiler, cann):
    destination, compiler, cann = (Path(p).resolve(strict=True) for p in (destination, compiler, cann))
    receipt = json.loads((destination / "provenance.json").read_text())
    for name, checksum in receipt["sources"].items():
        if Path(name).name != name or digest((destination / name).read_bytes()) != checksum:
            raise ValueError("staged fusion source changed")
    if receipt["binaries"] or any((destination / Path(n).with_suffix(".bin")).exists() for n in KERNELS):
        raise ValueError("preserve existing binary evidence; stage a fresh directory")
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    for name in KERNELS:
        binary = destination / Path(name).with_suffix(".bin")
        subprocess.run(
            [
                str(compiler),
                str(destination / name),
                str(binary),
                "--npu-arch=dav-2002",
                "-std=c++17",
                f"-I{destination}",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
            ],
            check=True,
        )
        receipt["binaries"][binary.name] = digest(binary.read_bytes())
    (destination / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--cann-root", type=Path)
    args = parser.parse_args()
    if bool(args.compiler) != bool(args.cann_root):
        parser.error("compiler and cann-root are required together")
    result = stage(args.source_root, args.output)
    if args.compiler:
        result = compile_staged(args.output, args.compiler, args.cann_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
