# SPDX-License-Identifier: Apache-2.0
"""Join an authenticated CUDA rebuild with unchanged clipped Ascend kernels."""

import argparse
import json
import shutil
from pathlib import Path

from .native_expert_probe import digest_file

NATIVE_GATE_FILES = frozenset({"glm_fused_gate_up.bin", "glm_fused_gate_up_w4.bin"})


def join(output, clipped_build, cuda_build):
    original = json.loads(clipped_build.read_bytes())
    rebuilt = json.loads(cuda_build.read_bytes())
    if original.get("swiglu_limit") != 10.0 or rebuilt.get("swiglu_limit") != 10.0:
        raise ValueError("joined kernels require the checkpoint's clipped activation contract")
    if set(original["native_gate_files"]) != NATIVE_GATE_FILES:
        raise ValueError("original clipped Ascend gate pair is incomplete")
    library = Path(rebuilt["library"])
    if digest_file(library) != rebuilt["library_sha256"]:
        raise ValueError("rebuilt CUDA library failed authentication")
    for name, checksum in original["native_gate_files"].items():
        if digest_file(clipped_build.parent / name) != checksum:
            raise ValueError("original clipped Ascend kernel failed authentication")
    output.mkdir(parents=True, exist_ok=False)
    for name in NATIVE_GATE_FILES:
        shutil.copyfile(clipped_build.parent / name, output / name)
    copied = output / library.name
    shutil.copyfile(library, copied)
    receipt = dict(
        original,
        cuda_library=str(copied.resolve()),
        cuda_library_sha256=rebuilt["library_sha256"],
        cuda_stream_build=rebuilt,
        original_clipped_build_sha256=digest_file(clipped_build),
        hardware_validated=False,
    )
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clipped-build", type=Path, required=True)
    parser.add_argument("--cuda-build", type=Path, required=True)
    args = parser.parse_args()
    join(args.output, args.clipped_build, args.cuda_build)


if __name__ == "__main__":
    main()
