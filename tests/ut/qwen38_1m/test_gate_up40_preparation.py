# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the candidate dispatch policy and verify guarded patch preparation."""

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.qwen4exp.prepare_gate_up40 import MATMUL, POLICY, POLICY_SOURCE, SCHEDULE, candidate_files, patch_text

ROOT = Path(__file__).resolve().parents[3]


def test_dispatch_changes_only_the_routed_31_to_40_gate_up_shape(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    (tmp_path / POLICY.name).write_text(POLICY_SOURCE)
    source = tmp_path / "policy.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#define __aicore__
#include "gate_up40_policy.h"
struct Data {
  int64_t metadataLanes, routed, numRows, nDim, kDim;
};
int main() {
  for (int64_t rows = 0; rows <= 25600; ++rows) {
    for (int64_t routed : {int64_t(0), int64_t(1)}) {
      Data td{8, routed, rows, 1280, 2560};
      assert(native_int4::UseExtendedGateUpSchedule(td) == (routed && rows >= 31 && rows <= 40));
      // The down shape, grouped/prefill and scalar-metadata paths remain unchanged.
      td.nDim = 2560; td.kDim = 640;
      assert(!native_int4::UseExtendedGateUpSchedule(td));
      td = Data{1, routed, rows, 1280, 2560};
      assert(!native_int4::UseExtendedGateUpSchedule(td));
    }
  }
}
""".replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>")
    )
    executable = tmp_path / "policy"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)], check=True
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.parametrize("profile", ["tile40", "stream40"])
def test_patch_applies_to_its_recorded_base_without_editing_main(tmp_path, profile):
    for path in (MATMUL, SCHEDULE):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / path).read_bytes())
    originals = {path: (tmp_path / path).read_bytes() for path in (MATMUL, SCHEDULE)}
    files = candidate_files(tmp_path, profile)
    patch = patch_text(tmp_path, files)
    subprocess.run(["git", "apply", "--check", "-"], input=patch, text=True, cwd=tmp_path, check=True)
    assert originals == {path: (tmp_path / path).read_bytes() for path in originals}
    subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=tmp_path, check=True)
    assert all((tmp_path / path).read_text() == content for path, content in files.items())
    if profile == "tile40":
        assert (tmp_path / SCHEDULE).read_bytes() == originals[SCHEDULE]
    else:
        # The default is still 30, including the existing down specialization.
        assert "uint32_t PIPELINED_ROUTE_LIMIT = 30>" in files[SCHEDULE]
        assert "MAX_COMBINED_ROUTES = 30" in files[SCHEDULE]
        assert "bool CACHE_GROUP_METADATA = false, uint32_t PIPELINED_ROUTE_LIMIT" in files[SCHEDULE]
        assert "Schedule<16, N, false, true, false, false, native_int4::EXTENDED_GATE_UP_ROUTE_LIMIT>" in files[MATMUL]


def test_stale_or_already_patched_source_is_rejected(tmp_path):
    for path in (MATMUL, SCHEDULE):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / path).read_bytes())
    files = candidate_files(tmp_path, "tile40")
    (tmp_path / MATMUL).write_text(files[MATMUL])
    with pytest.raises(ValueError, match="candidate source changed"):
        candidate_files(tmp_path, "tile40")


def test_preparation_retains_existing_60_route_down_capacity(tmp_path):
    for path in (MATMUL, SCHEDULE):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = (ROOT / path).read_text()
        destination.write_text(text.replace("MAX_COMBINED_ROUTES = 30", "MAX_COMBINED_ROUTES = 60"))
    files = candidate_files(tmp_path, "stream40")
    assert "MAX_COMBINED_ROUTES = 60" in files[SCHEDULE]
