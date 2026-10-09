# SPDX-License-Identifier: Apache-2.0
"""CPU admission and actual C++ scratch-layout checks for cached boundaries."""

import json
import subprocess

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.glm_fused_moe import NativeFusedMoE


@pytest.mark.parametrize("flag", [True, 1, "true"])
def test_incompatible_cache_build_creates_no_artifacts(tmp_path, flag):
    with pytest.raises(ValueError, match="expert boundary cache|must be boolean"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, cache_expert_ends=flag)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"cache_expert_ends": 1},
        {"cache_expert_ends": True},
        {"cache_expert_ends": True, "fused_moe": True, "prepared_weight_layout": True},
        {
            "cache_expert_ends": True,
            "fused_moe": True,
            "prepared_weight_layout": True,
            "vector_scale_products": True,
            "weight_decode_lut": True,
        },
    ],
)
def test_bad_cache_contract_does_not_load_kernels(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="expert boundary cache"):
        NativeFusedMoE(tmp_path, namespace="must_not_load", activation_bits=4)


def test_actual_cpp_layout_retains_row_indices_and_bounds_all_supported_expert_counts(tmp_path):
    source = tmp_path / "layout.cpp"
    source.write_text("""#include "glm_fused_route_cache.h"
#include <cassert>
template<unsigned Rows> void check() {
  using C = GlmFusedRouteCache::Layout<Rows>;
  static_assert(C::END_OFFSET >= 24576 + Rows * 4);
  static_assert(C::END_OFFSET + 288 * 8 <= 32768);
  static_assert(C::ENDS_PER_DMA == 4);
  for (unsigned experts = 1; experts <= C::MAX_EXPERTS; ++experts) {
    unsigned prefix = experts / C::ENDS_PER_DMA * C::ENDS_PER_DMA;
    assert(prefix <= experts);
    assert(prefix * 8 % 32 == 0);
    assert(experts - prefix <= 3);
    assert(C::END_OFFSET + experts * 8 <= C::MASK_BYTES);
  }
}
int main() { check<16>(); check<32>(); }
""")
    binary = tmp_path / "layout"
    subprocess.run(["c++", "-std=c++17", "-I" + str(builder.HERE), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    assert (builder.HERE / "glm_fused_route_cache.h").is_file()


def test_cli_records_cache_option(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build", "--build-dir", str(tmp_path), "--cache-expert-ends"])
    calls = []
    monkeypatch.setattr(builder, "build", lambda *args, **kwargs: calls.append(kwargs))
    builder.main()
    assert calls == [
        {
            "nz_prefill_min_rows": 0,
            "cache_expert_ends": True,
            "prefill_reduce_meta_cache": False,
            "direct_compact_down_scales": False,
            "group_major_input_scales": False,
            "group_major_down_scales": False,
        }
    ]
