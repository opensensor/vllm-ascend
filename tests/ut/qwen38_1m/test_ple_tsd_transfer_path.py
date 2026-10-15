# SPDX-License-Identifier: Apache-2.0
"""Guard private UDF transfer redirection against unrelated runtime uploads."""

import subprocess
from pathlib import Path


def test_private_udf_transfer_scope(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "scope.cpp"
    source.write_text(
        r"""
#include "ple_tsd_transfer_path.h"
#include <cassert>
int main() {
    const std::string base = "/home/HwHiAiUser/hdcd/device0";
    const std::string file = "0_Ascend-flow_model.tar.gz";
    std::string result;
    assert(ple_tsd_transfer_path("/private/" + file, base + "/123_" + file,
                                 base, base + "/aicpu", 123, result));
    assert(result == base + "/aicpu/123_" + file);
    assert(!ple_tsd_transfer_path("/private/" + file, base + "/124_" + file,
                                  base, base + "/aicpu", 123, result));
    assert(result.empty());
    assert(!ple_tsd_transfer_path("/private/" + file, base + "/123_" + file,
                                  base, base + "/other", 123, result));
    assert(!ple_tsd_transfer_path("/private/syskernels.tar.gz", base + "/123_" + file,
                                  base, base + "/aicpu", 123, result));
    assert(!ple_tsd_transfer_path("/private/" + file, base + "/../123_" + file,
                                  base, base + "/aicpu", 123, result));
    assert(!ple_tsd_transfer_path("/private/" + file, base + "/123_" + file,
                                  base, base + "/aicpu", 0, result));
    assert(!ple_tsd_transfer_path(file, base + "/123_" + file,
                                  base, base + "/aicpu", 123, result));
    assert(!ple_tsd_transfer_path("/private/" + file, "/123_" + file,
                                  "", "/aicpu", 123, result));
}
"""
    )
    binary = tmp_path / "scope"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "tools/qwen4exp"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)
