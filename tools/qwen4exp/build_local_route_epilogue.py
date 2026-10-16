# SPDX-License-Identifier: Apache-2.0
"""Stage private candidates from local git objects, without loading an NPU."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tools.qwen4exp.build_route_finalize import QUALIFIED_HEADERS

QUALIFIED_REVISION = "456e1ebcd"
HEADER_ROOT = "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"
ZERO_TAIL = """        if (expert + 1 == experts_) {
          auto out = output_.Get<half>();
          Duplicate(out, static_cast<half>(0), ELEMENTS);
          SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
          for (int64_t row = end; row < rows_; row += M) Store(out, tile, row, Min(M, rows_ - row));
          SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
"""
SOURCES = ("native_local_down_projection.cpp", "native_local_route_epilogue.cpp")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def local_output_header(source):
    """Change exactly one grouped zero-tail block in the audited header."""
    if sha256(source) != QUALIFIED_HEADERS["native_int4_schedule.h"]:
        raise ValueError("refuse a schedule other than the qualified frozen header")
    text = source.decode()
    if text.count(ZERO_TAIL) != 1:
        raise ValueError("qualified grouped tail-zero seam changed")
    return text.replace(ZERO_TAIL, "        // Inactive rows are private to the paired local epilogue.\n").encode()


def stage(root, destination, *, revision=QUALIFIED_REVISION):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    # Read and validate everything before creating an output directory. No
    # checkout, remote fetch, package import or device discovery is performed.
    headers = {
        name: subprocess.check_output(["git", "show", f"{revision}:{HEADER_ROOT}/{name}"], cwd=root)
        for name in QUALIFIED_HEADERS
    }
    if any(sha256(headers[name]) != expected for name, expected in QUALIFIED_HEADERS.items()):
        raise ValueError("local git revision does not contain the qualified headers")
    generated = local_output_header(headers["native_int4_schedule.h"])
    sources = {name: (root / "tools/qwen4exp" / name).read_bytes() for name in SOURCES}
    destination.mkdir(parents=True, exist_ok=False)
    files = {**headers, **sources, "native_int4_local_output.h": generated}
    for name, data in files.items():
        (destination / name).write_bytes(data)
    receipt = dict(
        schema_version=1,
        qualified_revision=subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True).strip(),
        sources={name: sha256(data) for name, data in files.items()},
        binaries={},
        device_opened=False,
        server_modified=False,
        hardware_validated=False,
        entries=dict(
            projection="qwen_local_down_projection_v1",
            ordered="qwen_local_route_finalize_v1",
            sequential="qwen_local_route_reduce_sequential_v1",
            pairwise="qwen_local_route_reduce_pairwise_v1",
        ),
    )
    (destination / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def compile_staged(destination, compiler, cann):
    """Explicit optional ACLRTC compile; no torch import, loader or install."""
    destination, compiler, cann = (Path(p).resolve(strict=True) for p in (destination, compiler, cann))
    receipt = json.loads((destination / "provenance.json").read_text())
    for name, checksum in receipt["sources"].items():
        if Path(name).name != name or sha256((destination / name).read_bytes()) != checksum:
            raise ValueError("staged source changed")
    if any((destination / Path(name).with_suffix(".bin")).exists() for name in SOURCES):
        raise ValueError("preserve existing binaries; stage a fresh directory")
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    for name in SOURCES:
        output = destination / Path(name).with_suffix(".bin")
        subprocess.run(
            [
                str(compiler),
                str(destination / name),
                str(output),
                "--npu-arch=dav-2002",
                "-std=c++17",
                f"-I{destination}",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
            ],
            check=True,
        )
        receipt["binaries"][output.name] = sha256(output.read_bytes())
    (destination / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--cann-root", type=Path)
    args = parser.parse_args()
    if bool(args.compiler) != bool(args.cann_root):
        parser.error("optional compilation requires both --compiler and --cann-root")
    result = stage(args.source_root, args.output)
    if args.compiler:
        result = compile_staged(args.output, args.compiler, args.cann_root)
    print(json.dumps(result, indent=2))
