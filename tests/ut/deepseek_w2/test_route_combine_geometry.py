# SPDX-License-Identifier: Apache-2.0
"""Compile the actual host geometry contract used by tiling and the adapter."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_route_combine_geometry_boundaries(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    include = Path(__file__).resolve().parents[3] / "csrc/gmm/w2_route_combine_v310"
    source = tmp_path / "geometry.cpp"
    source.write_text(r"""
#include <cassert>
#include "route_combine_geometry.h"
int main() {
    using NsW2Combine::ValidGeometry;
    assert(ValidGeometry(8, 32, 1, 8, 1));
    assert(ValidGeometry(10240, 4096, 1280, 8, 72));
    assert(ValidGeometry(32768, 16384, 1024, 32, 1024));
    assert(!ValidGeometry(0, 4096, 0, 8, 72));
    assert(!ValidGeometry(32769, 4096, 32769, 1, 72));
    assert(!ValidGeometry(8, 33, 1, 8, 72));
    assert(!ValidGeometry(8, 16416, 1, 8, 72));
    assert(!ValidGeometry(8, 4096, 1, 7, 72));
    assert(!ValidGeometry(33, 4096, 1, 33, 72));
    assert(!ValidGeometry(8, 4096, 1, 8, 0));
    assert(!ValidGeometry(8, 4096, 1, 8, 1025));
}
""")
    binary = tmp_path / "geometry"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Werror", "-I", str(include), str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)
