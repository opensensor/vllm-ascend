"""Build an isolated residual/RMSNorm/NZ-quantization candidate."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

NORM_ROWS = 8
NORM_INNER = 4096
NORM_ARGUMENTS = 8


def checked_build(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build.json").read_text())
    if (manifest.get("rows"), manifest.get("inner"), manifest.get("arguments")) != (
        NORM_ROWS,
        NORM_INNER,
        NORM_ARGUMENTS,
    ):
        raise ValueError("Unsupported norm/quant contract")
    if hashlib.sha256((directory / "norm-quant.bin").read_bytes()).hexdigest() != manifest["sha256"].get(
        "norm-quant.bin"
    ):
        raise ValueError("Norm/quant binary fingerprint mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True, help="Frozen compile-dense RTC helper")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent / "norm_quant.cpp"
    cann = args.cann.resolve()
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    binary = args.output / "norm-quant.bin"
    subprocess.run(
        [
            str(args.compiler),
            str(source),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
        ],
        check=True,
    )
    manifest = {
        "rows": NORM_ROWS,
        "inner": NORM_INNER,
        "arguments": NORM_ARGUMENTS,
        "sha256": {
            binary.name: hashlib.sha256(binary.read_bytes()).hexdigest(),
            source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }
    (args.output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
