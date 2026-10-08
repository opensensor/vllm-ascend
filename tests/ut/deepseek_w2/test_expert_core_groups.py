# SPDX-License-Identifier: Apache-2.0
"""Compile the device ownership planner on CPU and check exact tile coverage."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_expert_teams_cover_each_active_expert_tile_once(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    kernel_dir = Path(__file__).resolve().parents[3] / "csrc/gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel"
    source = tmp_path / "ownership.cpp"
    source.write_text(r"""
#include <cassert>
#include <vector>
#define __aicore__
#include "expert_core_groups.h"
int main() {
    // Match the serving contract: MTP1 verifies two rows per request,
    // each carrying eight routes. Every qualified concurrency retains W3
    // decode scheduling, including c3/c4 above the old 32-route cutoff.
    for (unsigned requests = 1; requests <= 4; ++requests)
        assert(!NsW2::AllowResidentW3Prefill(3, requests * 2 * 8));
    for (unsigned rows : {1, 8, 24, 32, 33, 48, 63, 64, 65, 72, 144, 5120, 20480}) {
        assert(NsW2::AllowResidentW3Prefill(2, rows));
        assert(NsW2::AllowResidentW3Prefill(4, rows));
        assert(NsW2::AllowResidentW3Prefill(3, rows) == (rows > 64));
    }
    for (unsigned cores = 1; cores <= 8; ++cores)
    for (unsigned requested = 0; requested <= 16; ++requested)
    for (unsigned rows : {8, 32, 255, 256, 5120})
    for (unsigned nTiles : {1, 2, 4, 16, 32, 128})
    for (unsigned pattern = 0; pattern < 5; ++pattern) {
        std::vector<unsigned> counts(72), writes(72 * nTiles);
        for (unsigned e = 0; e < counts.size(); ++e) {
            counts[e] = pattern == 0 ? 0 : pattern == 1 ? (e == 71 ? 257 : 0)
                       : pattern == 2 ? (e % 3 == 0 ? 129 : 0)
                       : pattern == 3 ? 1 : (e % 2 ? 257 : 18);
        }
        for (unsigned core = 0; core < cores; ++core) {
            const auto team = NsW2::MakeExpertCoreGroup(core, cores, requested, rows);
            assert(team.lanes > 0 && team.lane < team.lanes);
            assert(team.groups > 0 && team.group < team.groups);
            if (rows < 256 || requested == 0 || requested > cores || cores % requested)
                assert(team.groups == 1 && team.lanes == cores && team.lane == core);
            unsigned active = 0;
            for (unsigned e = 0; e < counts.size(); ++e) {
                if (!counts[e]) continue;
                if (active++ % team.groups != team.group) continue;
                for (unsigned n = team.lane; n < nTiles; n += team.lanes)
                    ++writes[e * nTiles + n];
            }
        }
        for (unsigned e = 0; e < counts.size(); ++e)
            for (unsigned n = 0; n < nTiles; ++n)
                assert(writes[e * nTiles + n] == (counts[e] ? 1u : 0u));
    }
}
""")
    executable = tmp_path / "ownership"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Werror", "-I", str(kernel_dir), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
