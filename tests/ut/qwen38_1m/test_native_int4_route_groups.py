# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the native INT4 route planner without loading an NPU runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest

KERNEL_DIR = Path(__file__).resolve().parents[3] / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"


def test_native_int4_route_groups_preserve_order_and_peer_rows(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")

    source = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#define __aicore__
#include "native_int4_route_groups.h"

using namespace native_int4;
struct RouteIds {
  const int32_t* ids;
  uint32_t* reads;
  int32_t GetValue(uint32_t row) {
    ++reads[row];
    return ids[row];
  }
};

template <bool ENABLE_DIRECT_LOOKUP>
void Check(const std::vector<int32_t>& ids, int64_t experts, uint32_t broadcastFactor) {
  std::array<int32_t, MAX_ROUTE_ROWS> expertIds{};
  std::array<uint32_t, MAX_ROUTE_ROWS> counts{}, groupOfRow{}, routeRows{}, sourceRows{}, reads{};
  std::array<uint32_t, MAX_ROUTE_ROWS + 1> starts{};
  std::vector<int32_t> expectedExperts;
  std::vector<std::vector<uint32_t>> expectedRows;
  for (uint32_t row = 0; row < ids.size(); ++row) {
    const int32_t id = ids[row];
    if (id < 0 || id >= experts) continue;
    uint32_t group = 0;
    while (group < expectedExperts.size() && expectedExperts[group] != id) ++group;
    if (group == expectedExperts.size()) {
      expectedExperts.push_back(id);
      expectedRows.emplace_back();
    }
    expectedRows[group].push_back(row);
  }
  const auto groups = BuildRouteGroups<ENABLE_DIRECT_LOOKUP>(
      RouteIds{ids.data(), reads.data()}, ids.size(), experts, broadcastFactor,
      expertIds.data(), counts.data(), groupOfRow.data(), starts.data(),
      routeRows.data(), sourceRows.data());
  assert(groups == expectedExperts.size());
  assert(starts[0] == 0);
  for (uint32_t row = 0; row < ids.size(); ++row) {
    assert(reads[row] == 1);
    assert(sourceRows[row] == row / broadcastFactor);
    if (ids[row] < 0 || ids[row] >= experts) {
      assert(groupOfRow[row] == MAX_ROUTE_ROWS);
    } else {
      assert(groupOfRow[row] < groups);
      assert(expertIds[groupOfRow[row]] == ids[row]);
    }
  }
  for (uint32_t group = 0; group < groups; ++group) {
    assert(expertIds[group] == expectedExperts[group]);
    assert(starts[group + 1] - starts[group] == expectedRows[group].size());
    assert(counts[group] == starts[group + 1]);
    for (uint32_t index = 0; index < expectedRows[group].size(); ++index)
      assert(routeRows[starts[group] + index] == expectedRows[group][index]);
  }
  assert(starts[groups] <= ids.size());
}

int main() {
  std::mt19937 random(310);
  for (uint32_t rows = 1; rows <= MAX_ROUTE_ROWS; ++rows) {
    for (const int64_t experts : {1, 3, 32, 128, 129, 512}) {
      for (uint32_t phase = 0; phase < 12; ++phase) {
        std::vector<int32_t> ids(rows);
        for (uint32_t row = 0; row < rows; ++row) {
          switch (phase) {
            case 0: ids[row] = 0; break;
            case 1: ids[row] = -1; break;
            case 2: ids[row] = row; break;
            case 3: ids[row] = row % 3; break;
            case 4: ids[row] = row % 2 == 0 ? std::numeric_limits<int32_t>::min()
                                               : std::numeric_limits<int32_t>::max(); break;
            default: ids[row] = static_cast<int32_t>(random() % 540) - 10;
          }
        }
        Check<true>(ids, experts, phase % 2 == 0 ? 1 : 10);
        Check<false>(ids, experts, phase % 2 == 0 ? 1 : 10);
      }
    }
  }
}
"""
    source_path = tmp_path / "native_int4_route_groups.cpp"
    source_path.write_text(source)
    executable = tmp_path / "native_int4_route_groups"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(KERNEL_DIR),
            str(source_path),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
