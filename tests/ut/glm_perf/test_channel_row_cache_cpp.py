# SPDX-License-Identifier: Apache-2.0
"""Production address preparation: exact token mapping and one GM read per row."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_route_row_cache_all_top_k_values_and_read_lifetime(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "kernel_operator.h").write_text(r"""
#pragma once
#include <cstdint>
#define __aicore__
namespace AscendC {
template<class T> struct GlobalTensor {
  T* values; unsigned* reads;
  T GetValue(unsigned n) const { ++*reads; return values[n]; }
};
template<class T> struct LocalTensor {
  T* values;
  void SetValue(unsigned n,T value) { values[n]=value; }
  T GetValue(unsigned n) const { return values[n]; }
};
}
""")
    source = tmp_path / "check.cpp"
    source.write_text(r"""
#include "glm_channel_row_cache.h"
#include <array>
int main() {
  for(unsigned count : {1u,2u,15u,32u,48u,64u}) for(int64_t topK=1;topK<=8;++topK) {
    std::array<int64_t,128> order;
    std::array<int32_t,80> cached;
    for(unsigned i=0;i<128;++i) order[i]=(int64_t(i)*7907)%65536;
    cached.fill(-909);
    unsigned reads=0;
    GlmChannelRowCache::Prepare({order.data(),&reads},{cached.data()+8},17,count,topK);
    if(reads!=count) return 1;
    // K loads and scale readbacks must reuse the prepared token addresses,
    // even if the original route storage is overwritten after preparation.
    auto original=order;
    order.fill(-1);
    for(unsigned k=0;k<32;++k) for(unsigned m=0;m<count;++m) {
      if(cached[8+m]!=original[17+m]/topK) return 2;
    }
    if(reads!=count) return 3;
    for(unsigned i=0;i<8;++i) if(cached[i]!=-909) return 4;
    for(unsigned i=8+count;i<cached.size();++i) if(cached[i]!=-909) return 5;
  }
}
""")
    binary = tmp_path / "check-cache"
    root = Path(__file__).resolve().parents[3] / "tools/glm_perf"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", f"-I{tmp_path}", f"-I{root}", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
