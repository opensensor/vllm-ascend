# SPDX-License-Identifier: Apache-2.0
"""Compile an isolated PLE selector, reusing a provenance-checked native bridge."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from tools.qwen4exp.ple_request import MAX_REQUESTS

ENTRY = "qwen_ple_request_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, output, cann, bridge_build):
    root, cann, bridge_build = (Path(path).resolve(strict=True) for path in (root, cann, bridge_build))
    previous = json.loads((bridge_build / "provenance.json").read_text())
    if previous["namespace"] != "qwen_checkpoint_preparation_v1" or previous["flush_host_queue"] is not False:
        raise ValueError("requires qualified owned native bridge without host queue drain")
    original_bridge = Path(previous["bridge"]["path"]).resolve(strict=True)
    if original_bridge.parent != bridge_build or digest(original_bridge) != previous["bridge"]["sha256"]:
        raise ValueError("native bridge provenance differs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for relative in (
        "tools/glm_perf/compile_reconstruction.cpp",
        "tools/qwen4exp/native_ple_request.cpp",
        "tools/qwen4exp/ple_request_integer.h",
    ):
        destination = output / Path(relative).name
        shutil.copy2(root / relative, destination)
        files[destination.name] = digest(destination)
    bridge = output / original_bridge.name
    shutil.copy2(original_bridge, bridge)
    compiler = output / "compile-ple-request"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(output / "compile_reconstruction.cpp"),
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
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binary = output / "native_ple_request.bin"
    subprocess.run(
        [
            str(compiler),
            str(output / "native_ple_request.cpp"),
            str(binary),
            "--npu-arch=dav-2002",
            f"-I{output}",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    manifest = dict(
        entry=ENTRY,
        max_requests=MAX_REQUESTS,
        namespace=previous["namespace"],
        sources=files,
        bridge=dict(path=str(bridge), sha256=digest(bridge)),
        binary=dict(path=str(binary), sha256=digest(binary)),
        flush_host_queue=False,
        hardware_validated=False,
        server_modified=False,
    )
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "cann", "bridge-build"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, args.cann, args.bridge_build), indent=2))
