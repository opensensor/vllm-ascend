# SPDX-License-Identifier: Apache-2.0
"""Exercise the bounded receipt formatter without Ascend runtime dependencies."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_bounded_probe_receipt(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires C++ compiler")
    source = tmp_path / "test.cpp"
    source.write_text(r"""
#include "ple_probe_text.h"
#include <cassert>
#include <cstring>
int main() {
    char bytes[10];
    std::memset(bytes, 'X', sizeof(bytes));
    PleProbeText zero(bytes, 0);
    assert(!zero.append("abc") && zero.size() == 0 && bytes[0] == 'X');
    PleProbeText one(bytes, 1);
    assert(!one.append("a") && bytes[0] == 0 && bytes[1] == 'X');
    PleProbeText exact(bytes, 5);
    assert(exact.append("%s%d", "abc", 1));
    assert(exact.size() == 4 && !exact.truncated());
    assert(!exact.append("overflow") && exact.truncated());
    assert(std::strcmp(bytes, "abc1") == 0 && bytes[5] == 'X');
    PleProbeText overflow(bytes, 7);
    assert(overflow.append("ok"));
    assert(!overflow.append("%u", 123456U));
    assert(overflow.size() == 6 && overflow.truncated());
    assert(std::strcmp(bytes, "ok1234") == 0 && bytes[7] == 'X');
    assert(!overflow.append("%10000d", 1));
}
""")
    include = Path(__file__).resolve().parents[3] / "tools/qwen4exp"
    binary = tmp_path / "test"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{include}", str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
