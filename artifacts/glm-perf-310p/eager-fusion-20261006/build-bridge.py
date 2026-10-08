# SPDX-License-Identifier: Apache-2.0
"""Build a separately registered resident bridge; never replaces loaded code."""

from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def main():
    root = Path(__file__).resolve().parent
    cann = Path("/usr/local/Ascend/cann-9.1.0")
    npu = Path(torch_npu.__file__).parent
    load(
        name="glm_eager_fusions_v1",
        sources=[str(root / "bridge.cpp")],
        extra_include_paths=[str(npu / "include"), str(cann / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu", f"-L{cann / 'lib64'}", "-lascendcl"],
        build_directory=str(root),
        is_python_module=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
