"""Compile isolated input quantization without replacing the live matrix build."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tools.rtsp_scene.input_quant import INPUT_WIDTHS, MAX_ROWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    cann = args.cann.resolve()
    compiler = args.output / "compile-input"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(source / "compile_dense.cpp"),
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
    hashes = {}
    for inner in INPUT_WIDTHS:
        binary = args.output / f"input-k{inner}.bin"
        subprocess.run(
            [
                str(compiler),
                str(source / "input_quant.cpp"),
                str(binary),
                "--npu-arch=dav-2002",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
                f"-DRTSP_QUANT_INNER={inner}",
            ],
            check=True,
        )
        hashes[binary.name] = hashlib.sha256(binary.read_bytes()).hexdigest()
    for name in ("input_quant.cpp", "build_input_quant.py", "compile_dense.cpp"):
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    (args.output / "build.json").write_text(
        json.dumps(
            {
                "max_rows": MAX_ROWS,
                "arguments": 4,
                "sha256": hashes,
                "cann_root": str(cann),
                "architecture": "dav-2002",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
