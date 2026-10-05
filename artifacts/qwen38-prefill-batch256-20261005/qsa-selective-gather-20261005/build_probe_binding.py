# SPDX-License-Identifier: Apache-2.0
"""Build only the additional QSA gather binding against a qualified source tree."""

import argparse
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    common = args.source_root / "csrc/aclnn_torch_adapter"
    npu_root = Path(torch_npu.__file__).resolve().parent
    args.build_dir.mkdir(parents=True, exist_ok=True)
    result = load(
        name="qsa_selective_probe",
        sources=[
            str(experiment / "probe_binding.cpp"),
            str(common / "NPUBridge.cpp"),
            str(common / "NPUStorageImpl.cpp"),
        ],
        extra_include_paths=[str(common), str(args.cann_root / "include"), str(npu_root / "include")],
        extra_cflags=["-O3", "-DASCEND_PLATFORM_310P", "-fvisibility=hidden"],
        extra_ldflags=[
            f"-L{npu_root / 'lib'}",
            f"-L{args.cann_root / 'lib64'}",
            "-ltorch_npu",
            "-lascendcl",
            "-lopapi",
            "-ldl",
        ],
        build_directory=str(args.build_dir),
        is_python_module=False,
        verbose=True,
    )
    print(result)


if __name__ == "__main__":
    main()
