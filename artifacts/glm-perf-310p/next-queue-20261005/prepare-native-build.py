# SPDX-License-Identifier: Apache-2.0
"""Generate a fresh build directory; never load code or contact an NPU.

Use on the qualified build host. The generated build.sh compiles on CPU only.
It deliberately does not generate a load manifest before numerical validation.
"""

import argparse
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    previous = root.parent / "decode-fusions-resident-20261005"
    (output / "bridge.cpp").write_text(
        (previous / "bridge.cpp").read_text().replace("glm_rotation_v1", "glm_kda_prepare_v1")
    )
    (output / "kda-gate-beta.cpp").write_bytes((root / "kda-gate-beta.cpp").read_bytes())
    (output / "compile-kernel.cpp").write_bytes(
        (root.parent / "native-resident-20261005/compile-kernel.cpp").read_bytes()
    )
    ninja = (previous / "build.ninja").read_text()
    ninja = ninja.replace("glm_rotation_bridge_v1", "glm_kda_prepare_bridge_v1")
    ninja = ninja.replace(
        "/home/matteius/experiments/glm-decode-fusions-resident-20261005/bridge.cpp", str(output / "bridge.cpp")
    )
    (output / "build.ninja").write_text(ninja)
    (output / "build.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd -- {shlex.quote(str(output))}\n"
        "test ! -e kda-gate-beta-v1.bin\ntest ! -e glm_kda_prepare_bridge_v1.so\n"
        "c++ compile-kernel.cpp -I/usr/local/Ascend/cann-9.1.0/include "
        "-L/usr/local/Ascend/cann-9.1.0/lib64 -lacl_rtc -lascendcl -o compile-kernel\n"
        "./compile-kernel kda-gate-beta.cpp kda-gate-beta-v1.bin\nninja -f build.ninja\n"
    )
    print(output / "build.sh")


if __name__ == "__main__":
    main()
