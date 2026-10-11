# SPDX-License-Identifier: Apache-2.0
"""Execute the production repeat-stride epilogue with a CPU vector interpreter."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_production_broadcast_header_all_rows_and_column_halves(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "kernel_operator.h").write_text(r"""
#pragma once
#include <cstdint>
#define __aicore__
namespace AscendC {
enum Pipe { PIPE_V };
template<Pipe> void PipeBarrier() {}
template<class T> struct LocalTensor {
  T* data;
  LocalTensor operator[](unsigned n) const { return {data+n}; }
};
struct BinaryRepeatParams { unsigned d, a, b, dr, ar, br; };
template<class T> void Muls(LocalTensor<T> d, LocalTensor<T> a, T v, unsigned n) {
  for (unsigned i=0;i<n;++i) d.data[i]=a.data[i]*v;
}
template<class T> void Add(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, unsigned n) {
  for (unsigned i=0;i<n;++i) d.data[i]=a.data[i]+b.data[i];
}
inline void Mul(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, unsigned n) {
  for (unsigned i=0;i<n;++i) d.data[i]=a.data[i]*b.data[i];
}
template<bool multiply,class T> void Binary(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b,
                                  uint64_t mask, unsigned count, BinaryRepeatParams p) {
  for (unsigned r=0;r<count;++r) for (unsigned i=0;i<mask;++i) {
    unsigned di=r*p.dr*8+(i/8)*p.d*8+i%8;
    unsigned ai=r*p.ar*8+(i/8)*p.a*8+i%8;
    unsigned bi=r*p.br*8+(i/8)*p.b*8+i%8;
    d.data[di]=multiply ? a.data[ai]*b.data[bi] : a.data[ai]+b.data[bi];
  }
}
template<class T> void Add(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b,
                uint64_t mask, unsigned count, BinaryRepeatParams p) { Binary<false>(d,a,b,mask,count,p); }
inline void Mul(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,
                uint64_t mask, unsigned count, BinaryRepeatParams p) { Binary<true>(d,a,b,mask,count,p); }
}
""")
    source = tmp_path / "check.cpp"
    source.write_text(r"""
#include "glm_channel_broadcast.h"
#include "glm_channel_integer.h"
#include <vector>
#include <cstdio>
int main() {
  for(unsigned count : {1u,2u,15u,18u,32u,48u,64u}) for(unsigned group : {256u,512u,1024u,2048u,4096u}) {
    std::vector<int32_t> low(count*128+16,991), high(count*128+16,992), sums(128), expected(count*128);
    for(unsigned c=0;c<128;++c) sums[c]=(int(c)%17-8)*int(group);
    for(unsigned r=0;r<count;++r) for(unsigned c=0;c<128;++c) {
      low[r*128+c]=(int(r+c)%17-8)*8*int(group);
      high[r*128+c]=(int(r*3+c)%17-8)*8*int(group);
      int64_t oracle=int64_t(low[r*128+c])+16*int64_t(high[r*128+c])+8*int64_t(sums[c]);
      float previous=float(low[r*128+c])+16.0f*float(high[r*128+c]);
      previous+=8.0f*float(sums[c]);
      if(float(oracle)!=previous) return 3;
      expected[r*128+c]=int32_t(oracle);
    }
    GlmChannelInteger::ReconstructA8({low.data()},{high.data()},{sums.data()},count);
    for(unsigned i=0;i<expected.size();++i) if(low[i]!=expected[i]) return 4;
    for(unsigned i=expected.size();i<low.size();++i) if(low[i]!=991 || high[i]!=992) return 5;
  }
  for(unsigned count : {1u,2u,15u,18u,32u,48u,64u}) for(unsigned bits : {4u,8u}) {
    std::vector<float> values(count*128+16,991), original(values), weights(128), correction(128), scales(count*8);
    for(unsigned c=0;c<128;++c) { weights[c]=float(c+1)/257; correction[c]=float(int(c)-64)*8; }
    for(unsigned r=0;r<count;++r) {
      for(unsigned c=0;c<128;++c) values[r*128+c]=float(int(r*128+c)%19-9)*16;
      for(unsigned lane=0;lane<8;++lane) scales[r*8+lane]=float(r+1)/127;
    }
    original=values;
    GlmChannelBroadcast::ScaleRows({values.data()},{weights.data()},{correction.data()},{scales.data()},count,bits);
    for(unsigned r=0;r<count;++r) for(unsigned c=0;c<128;++c) {
      float expected=original[r*128+c];
      if(bits==8) expected+=correction[c];
      expected*=weights[c]; expected*=scales[r*8];
      if(values[r*128+c]!=expected) { std::fprintf(stderr,"rows=%u bits=%u r=%u c=%u\n",count,bits,r,c); return 1; }
    }
    for(unsigned i=count*128;i<values.size();++i) if(values[i]!=991) return 2;
  }
}
""")
    root = Path(__file__).resolve().parents[3] / "tools/glm_perf"
    binary = tmp_path / "check-broadcast"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{root}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
