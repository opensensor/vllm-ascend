# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the production row-window planner and SwiGLU shape contract."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_resident_windows_and_swiglu_boundaries(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires C++ compiler")
    root = Path(__file__).resolve().parents[3] / "csrc/gmm"
    source = tmp_path / "geometry.cpp"
    source.write_text(r"""
#include <cassert>
#include <vector>
#define __aicore__
#include "w2_blocked_dequant_matmul_v310/op_kernel/row_reuse_geometry.h"
#include "w2_swiglu_v310/swiglu_geometry.h"
int main() {
    using namespace NsW2Rows;
    for (unsigned rows : {0,1,16,127,128,129,255,256,257,383,511,512,513,2560,20480,32768}) {
        std::vector<unsigned> coverage(rows);
        unsigned windows = 0;
        for (unsigned first = 0; first < rows; first += WINDOW_ROWS) {
            ++windows;
            unsigned visited = 0;
            for (unsigned tile = 0; tile < TILES; ++tile) {
                const unsigned count = TileRows(rows, first, tile);
                assert(count <= TILE_ROWS);
                for (unsigned i = 0; i < count; ++i) {
                    assert(first + tile * TILE_ROWS + i < rows);
                    ++coverage[first + tile * TILE_ROWS + i];
                }
                visited += count;
            }
            assert(visited == WindowRows(rows, first));
            assert(TileRows(rows, first, TILES) == 0);
        }
        for (auto count : coverage) assert(count == 1);
        assert(windows == (rows + WINDOW_ROWS - 1) / WINDOW_ROWS);
        assert(WindowRows(rows, rows) == 0);
        assert(WindowRows(rows, rows + 1) == 0);
    }
    using NsW2Swiglu::ValidGeometry;
    for (int rows : {1,8,32,5120,10240,20480,32768})
        for (int width : {64,2112,4096,8192,32768}) assert(ValidGeometry(rows, width));
    for (int rows : {-1,0,32769}) assert(!ValidGeometry(rows, 4096));
    for (int width : {-1,0,32,63,65,32770,32832}) assert(!ValidGeometry(8, width));
}
""")
    binary = tmp_path / "geometry"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Werror", "-I", str(root), str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)
