# SPDX-License-Identifier: Apache-2.0
"""Regress DMA overwriting a live Gather table and storing unfinished output."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_channel_dma_dependencies_preserve_live_vector_reads_and_publish_stores(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "kernel_operator.h").write_text(r"""
#pragma once
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
#include <cstring>
#define __aicore__
namespace AscendC {
enum class HardEvent {V_MTE2, MTE2_V, V_MTE3, MTE3_V};
enum Pipe {PIPE_ALL};
constexpr unsigned EVENT_ID0=0;
std::vector<std::function<void()>> vectorWork, incoming;
bool events[4]{};
void flush(std::vector<std::function<void()>>& work){for(auto& f:work)f();work.clear();}
template<HardEvent E>void SetFlag(unsigned slot){assert(slot==0&&!events[int(E)]);events[int(E)]=true;}
template<HardEvent E>void WaitFlag(unsigned slot){
 assert(slot==0&&events[int(E)]);events[int(E)]=false;
 if(E==HardEvent::V_MTE2||E==HardEvent::V_MTE3)flush(vectorWork);
 if(E==HardEvent::MTE2_V)flush(incoming);
}
template<Pipe>void PipeBarrier(){}
template<class T>struct LocalTensor{T* data;};
template<class T>struct GlobalTensor{T* data;};
template<class T>void DataCopy(LocalTensor<T>d,GlobalTensor<T>s,unsigned count){
 incoming.push_back([=](){std::memcpy(d.data,s.data,count*sizeof(T));});
}
template<class T>void DataCopy(GlobalTensor<T>d,LocalTensor<T>s,unsigned count){
 std::memcpy(d.data,s.data,count*sizeof(T));
}
}
""")
    source = tmp_path / "check.cpp"
    source.write_text(r"""
#include "glm_channel_dma.h"
using namespace AscendC;
int main(){
 for(int step=0;step<20;++step){
  int table[4]={0,2,4,6}, replacement[4]={1,3,5,7};
  int products[8]={-10,20,-30,40,-50,60,-70,80}, gathered[4]={}, output[4]={};
  vectorWork.push_back([&](){for(int i=0;i<4;++i)gathered[i]=products[table[i]];});
  // The previous table must remain valid until the vector Gather consumes it.
  GlmChannelDma::CopyIn(LocalTensor<int>{table},GlobalTensor<int>{replacement},4);
  assert(gathered[0]==-10&&gathered[3]==-70&&table[0]==1&&table[3]==7);
  vectorWork.push_back([&](){for(int i=0;i<4;++i)gathered[i]=products[table[i]]*2;});
  GlmChannelDma::WriteReady();
  DataCopy(GlobalTensor<int>{output},LocalTensor<int>{gathered},4);
  GlmChannelDma::Fence();
  assert(output[0]==40&&output[3]==160);
  assert(vectorWork.empty()&&incoming.empty());
  for(bool event:events)assert(!event);
 }
}
""")
    root = Path(__file__).resolve().parents[3] / "tools/glm_perf"
    binary = tmp_path / "check-dma"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", f"-I{tmp_path}", f"-I{root}", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
