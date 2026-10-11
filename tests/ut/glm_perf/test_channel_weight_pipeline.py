# SPDX-License-Identifier: Apache-2.0
"""Execute the production K/group loop with strict consumable event tokens."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_production_weight_slots_event_balance_group_accumulation_and_repeated_blocks(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (Path(__file__).resolve().parents[3] / "tools/glm_perf/glm_channel_projection.cpp").read_text()
    block = source[
        source.index("  __aicore__ inline void ProjectBlock(") : source.index("  __aicore__ inline int64_t SourceRow(")
    ]
    prefetch = source[
        source.index("  __aicore__ inline void PrefetchWeight(") : source.index("  __aicore__ inline void Readback(")
    ]
    # The source's closing #endif belongs to its declaration guard; open it here.
    text = (
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define GLM_CHANNEL_STREAM_WEIGHTS
constexpr unsigned ROWS=64, N=128, K=256, K_FRAGMENT=64, NZ_N=16, LANES=8;
constexpr unsigned B_BYTES=N*K/2, ELEMENTS=ROWS*N;
constexpr unsigned EVENT_ID0=0, EVENT_ID2=2;
using half=float;
using int4b_t=int8_t;
enum class HardEvent { MTE1_MTE2, M_MTE1, MTE2_MTE1, MTE1_M };
bool events[4][8]{};
template<HardEvent E> void SetFlag(unsigned slot) {
  assert(!events[int(E)][slot]); events[int(E)][slot]=true;
}
template<HardEvent E> void WaitFlag(unsigned slot) {
  assert(events[int(E)][slot]); events[int(E)][slot]=false;
}
enum Pipe {PIPE_V,PIPE_ALL};
template<Pipe> void PipeBarrier() {}
template<class T> struct Tensor {
  T* data;
  Tensor operator[](unsigned i) {return {data+i};}
  template<class U> Tensor<U> ReinterpretCast() {return {reinterpret_cast<U*>(data)};}
};
struct Buffer {
  std::vector<uint64_t> data=std::vector<uint64_t>(16384);
  template<class T> Tensor<T> Get() {return {reinterpret_cast<T*>(data.data())};}
};
enum class RoundMode {CAST_NONE};
template<class T> void Duplicate(Tensor<T> d, T v, unsigned n) {for(unsigned i=0;i<n;++i)d.data[i]=v;}
template<class D,class S> void Cast(Tensor<D>d,Tensor<S>s,RoundMode,unsigned n){
 for(unsigned i=0;i<n;++i)d.data[i]=s.data[i];}
template<class T> void DataCopy(Tensor<T>d,Tensor<T>s,unsigned n){std::memcpy(d.data,s.data,n*sizeof(T));}
struct LoadData2DParams {unsigned repeatTimes,srcStride;bool ifTranspose;};
void LoadData(Tensor<int8_t>d,Tensor<int8_t>s,LoadData2DParams){DataCopy(d,s,B_BYTES);}
struct MmadParams {unsigned m,n,k;bool cmatrixInitVal;};
void Mmad(Tensor<int32_t>c,Tensor<int8_t>,Tensor<int8_t>b,MmadParams p){
  if(p.cmatrixInitVal)c.data[0]=0;
  c.data[0]+=b.data[0];
}
struct Projection {
  Buffer a2_,b1_,b2_,c_,accum_,outputUb_;
  Tensor<int8_t> codes_;
  unsigned k_,group_,n_=256,bits_,groupsRead=0;
  void LoadActivation(int64_t,unsigned,unsigned,unsigned){WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);}
  void Readback(int64_t,unsigned,unsigned,int64_t expert,int64_t tile,unsigned group){
    int expected=0;
    for(unsigned kb=group*group_/K;kb<(group+1)*group_/K;++kb)
      expected+=codes_.data[((expert*(n_/N)+tile)*(k_/K)+kb)*B_BYTES];
    assert(c_.Get<int32_t>().data[0]==expected); ++groupsRead;
  }
"""
        + block
        + "\n#ifdef GLM_CHANNEL_STREAM_WEIGHTS\n"
        + prefetch
        + r"""
};
int main(){
 for(unsigned k:{256,512,1024,4096}) for(unsigned group:{256,512,1024}) {
  if(group>k || k%group)continue;
  std::vector<int8_t> weights(4*(k/K)*B_BYTES);
  for(unsigned i=0;i<4*(k/K);++i) weights[i*B_BYTES]=1+i%16;
  for(unsigned bits:{4,8})for(unsigned rows:{1,7,16,32}) {
   Projection p; p.k_=k;p.group_=group;p.bits_=bits;p.codes_={weights.data()};
   for(unsigned expert=0;expert<2;++expert)for(unsigned tile=0;tile<2;++tile){
    p.ProjectBlock(0,rows,expert,tile);
    for(auto& pipe:events)for(bool outstanding:pipe)assert(!outstanding);
   }
   assert(p.groupsRead==4*k/group);
  }
 }
}
"""
    )
    path, binary = tmp_path / "check-stream.cpp", tmp_path / "check-stream"
    path.write_text(text)
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(path), "-o", str(binary)], check=True, capture_output=True, text=True
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
