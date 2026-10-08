# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the staged KDA patch, including its actual loop arithmetic.

The small AscendC simulation does not validate device event ordering, vector
approximations, or UB allocation; those still require native build/NPU gates.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "artifacts/glm-perf-310p/next-queue-20261005"


@pytest.fixture
def sources(tmp_path):
    metadata = json.loads((STUDY / "kda-prepare-head-base.json").read_text())
    original = (ROOT / metadata["path"]).read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == metadata["sha256"]
    target = tmp_path / metadata["path"]
    target.parent.mkdir(parents=True)
    target.write_text(original)
    subprocess.run(["git", "apply", str(STUDY / "kda-prepare-head.patch")], cwd=tmp_path, check=True)
    return original, target.read_text()


def compiler():
    result = shutil.which("c++")
    if result is None:
        pytest.skip("requires host C++ compiler")
    return result


def test_disabled_flag_preserves_preprocessed_kernel(sources, tmp_path):
    (tmp_path / "kernel_operator.h").write_text("")
    outputs = []
    for name, source in zip(("baseline", "candidate"), sources):
        path = tmp_path / f"{name}.cpp"
        path.write_text(source)
        result = subprocess.run(
            [compiler(), "-E", "-P", "-x", "c++", "-I", str(tmp_path), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout.split())
    assert outputs[0] == outputs[1]


def method(source, name):
    start = source.index("    __aicore__ inline void " + name + "(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_actual_gate_and_chunk_loops_match_with_head_reuse(sources, tmp_path):
    prelude = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define GLM_KDA_GATE_PREPARE_HEAD
constexpr float RCP_LN2 = 1.4426950408889634f;
constexpr int PIPE_V=0;
constexpr int GATE_MTE2_V_EVENT_ID=0, GATE_V_MTE3_EVENT_ID=1;
constexpr int GATE_MTE3_MTE2_EVENT_ID=2, GATE_MTE3_V_EVENT_ID=5;
enum class HardEvent {MTE2_V, V_MTE3, MTE3_MTE2, MTE3_V};
template<int> void PipeBarrier() {}
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {}
template<class T> using GlobalTensor=std::vector<T>;
template<class T> struct LocalTensor {
    std::vector<T>* data;
    T& operator[](unsigned i) { return (*data)[i]; }
};
struct Buffer {
    std::vector<float> data=std::vector<float>(256);
    template<class T> LocalTensor<T> Get() { return {&data}; }
};
void Duplicate(LocalTensor<float> a,float v,unsigned n) { for(unsigned i=0;i<n;++i)a[i]=v; }
void Add(LocalTensor<float> a,LocalTensor<float>b,LocalTensor<float>c,unsigned n) {
    for(unsigned i=0;i<n;++i)a[i]=b[i]+c[i];
}
void Muls(LocalTensor<float>a,LocalTensor<float>b,float c,unsigned n) {
    for(unsigned i=0;i<n;++i)a[i]=b[i]*c;
}
void Exp(LocalTensor<float>a,LocalTensor<float>b,unsigned n) {
    for(unsigned i=0;i<n;++i)a[i]=std::exp(b[i]);
}
void Adds(LocalTensor<float>a,LocalTensor<float>b,float c,unsigned n) {
    for(unsigned i=0;i<n;++i)a[i]=b[i]+c;
}
void Div(LocalTensor<float>a,LocalTensor<float>b,LocalTensor<float>c,unsigned n) {
    for(unsigned i=0;i<n;++i)a[i]=b[i]/c[i];
}
"""
    members = r"""
    uint64_t batch_=1,t_=0,hv_=0,k_=0,chunkSize_=64,maxChunks_=0,seqNum_=0;
    bool hasCuSeqlens_=false,hasALog_=true,hasDtBias_=true;
    float lowerBound_=-5,headExpA_=1;
    unsigned expCalls=0,biasCopies=0;
    Buffer accBuf_,rowBuf_,tmpBuf_,oneBuf_,headBiasBuf_;
    GlobalTensor<float> g_,gk_,aLog_,dtBias_;
    GlobalTensor<int64_t> cuSeqlens_;
    uint64_t Offset(uint64_t b,uint64_t t,uint64_t h,uint64_t k) {
        return ((b*t_+t)*hv_+h)*k_+k;
    }
    void LoadGateRow(uint64_t offset,LocalTensor<float>& row) {
        for(unsigned k=0;k<k_;++k)row[k]=g_[offset+k];
    }
    void CopyFloatVectorIn(LocalTensor<float>& dst,GlobalTensor<float>& src,uint64_t off,uint64_t n) {
        ++biasCopies; for(unsigned i=0;i<n;++i)dst[i]=src[off+i];
    }
    void CopyFloatVectorOut(GlobalTensor<float>& dst,uint64_t off,LocalTensor<float>& src,uint64_t n) {
        for(unsigned i=0;i<n;++i)dst[off+i]=src[i];
    }
    float ReadFloat(GlobalTensor<float>& src,uint64_t off) { return src[off]; }
    int64_t ReadInt64(GlobalTensor<int64_t>& src,uint64_t off) { return src[off]; }
    float ExpScalar(float x) { ++expCalls; return std::exp(x); }
    void Setup(unsigned tokens,unsigned heads,unsigned channels,bool varlen,std::vector<int64_t> bounds) {
        t_=tokens; hv_=heads; k_=channels; maxChunks_=(t_+63)/64;
        hasCuSeqlens_=varlen; cuSeqlens_=bounds; seqNum_=bounds.size()-1;
        g_.resize(tokens*heads*channels); gk_.resize(g_.size());
        aLog_.resize(heads); dtBias_.resize(heads*channels);
        for(unsigned i=0;i<g_.size();++i)g_[i]=std::sin(float(i))*10;
        for(unsigned h=0;h<heads;++h)aLog_[h]=float(h)/heads-0.5f;
        for(unsigned i=0;i<dtBias_.size();++i)dtBias_[i]=std::cos(float(i));
    }
    void Run() {
        unsigned tasks=hasCuSeqlens_?seqNum_*hv_:batch_*hv_*maxChunks_;
        for(unsigned t=0;t<tasks;++t)ProcessTask(t);
    }
"""
    classes = []
    for name, source in zip(("Baseline", "Candidate"), sources):
        methods = [method(source, member) for member in ("ApplyGate", "ProcessTask", "ProcessChunk")]
        if name == "Candidate":
            methods.insert(0, method(source, "PrepareHead"))
        classes.append(f"template<bool SAFE_GATE> struct {name} {{\n" + members + "\n".join(methods) + "\n};\n")
    main = r"""
template<bool SAFE> void Check(unsigned tokens,unsigned heads,unsigned channels,bool varlen,
                              std::vector<int64_t> bounds) {
    for(bool bias : {false,true}) for(bool alog : {false,true}) {
        Baseline<SAFE> baseline; Candidate<SAFE> candidate;
        baseline.Setup(tokens,heads,channels,varlen,bounds);
        candidate.Setup(tokens,heads,channels,varlen,bounds);
        baseline.hasDtBias_=candidate.hasDtBias_=bias;
        baseline.hasALog_=candidate.hasALog_=alog;
        baseline.Run(); candidate.Run();
        assert(baseline.gk_.size()==candidate.gk_.size());
        if(!baseline.gk_.empty())
            assert(std::memcmp(baseline.gk_.data(),candidate.gk_.data(),baseline.gk_.size()*4)==0);
        unsigned tasks=0;
        if(varlen) { for(unsigned i=1;i<bounds.size();++i)tasks+=(bounds[i]>bounds[i-1])*heads; }
        else tasks=((tokens+63)/64)*heads;
        assert(baseline.expCalls==(SAFE&&alog?tokens*heads:0));
        assert(candidate.expCalls==(SAFE&&alog?tasks:0));
        assert(candidate.biasCopies==(SAFE&&bias?tasks:0));
    }
}
int main() {
    for(bool varlen : {false,true}) {
        Check<true>(134,3,4,varlen,{0,0,1,67,134});
        Check<false>(134,3,4,varlen,{0,0,1,67,134});
        Check<true>(640,16,128,varlen,{0,640});
        Check<true>(0,2,128,varlen,{0,0});
    }
}
"""
    path = tmp_path / "simulate.cpp"
    path.write_text(prelude + "\n".join(classes) + main)
    binary = tmp_path / "simulate"
    subprocess.run([compiler(), "-std=c++17", "-O2", "-ffp-contract=off", str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
