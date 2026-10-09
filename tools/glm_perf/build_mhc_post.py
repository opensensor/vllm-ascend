# SPDX-License-Identifier: Apache-2.0
"""Compile append-only FP16/FP32-input mHC experiments, without opening a device."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from tools.glm_perf.build_reconstruction import build

HERE = Path(__file__).resolve().parent


def build_mhc(build_dir, cann_root, source_root, version, *, finish_only=False, round_state=True):
    if not round_state and finish_only:
        raise ValueError("final FP32 mixer must replace the complete einsum")
    build(
        build_dir,
        cann_root,
        source_root,
        version=version,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
    )
    cann = cann_root.resolve(strict=True)
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    frozen = build_dir / "mhc_post_kernel.cpp"
    shutil.copy2(HERE / frozen.name, frozen)
    helper = build_dir / "mhc_post_native.py"
    shutil.copy2(HERE / helper.name, helper)
    records = {}
    for bits in (16, 32):
        binary = build_dir / f"mhc_post_fp{bits}.bin"
        command = [
            str(build_dir / "compile-reconstruction"),
            str(frozen),
            str(binary),
            "--npu-arch=dav-2002",
            f"--sysroot={cann / 'tools/hcc/sysroot'}",
            f"-isystem{target}",
            f"-isystem{target / 'aarch64-target-linux-gnu'}",
            *(["-DGLM_MHC_FP32_INPUT"] if bits == 32 else []),
            *(["-DGLM_MHC_FINISH_ONLY"] if finish_only else []),
            *(["-DGLM_MHC_NO_STATE_ROUNDING"] if not round_state else []),
        ]
        subprocess.run(command, check=True)
        records[binary.name] = {"sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "command": command}
    provenance = {
        "namespace": f"glm_reconstruction_v{version}",
        "version": version,
        "source_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "binaries": records,
        "state_rounding": "fp16_in_fp32_storage" if round_state else "none_fp32",
        "fp32_input_preserved": True,
        "finish_only": finish_only,
    }
    (build_dir / "mhc-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return build_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--source-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--finish-only", action="store_true", help="retain the reference einsum; fuse its epilogue")
    parser.add_argument("--no-state-rounding", action="store_true", help="experimental final mixer with FP32 output")
    args = parser.parse_args()
    print(
        build_mhc(
            args.build_dir.resolve(),
            args.cann_root,
            args.source_root,
            args.version,
            finish_only=args.finish_only,
            round_state=not args.no_state_rounding,
        )
    )


if __name__ == "__main__":
    main()
