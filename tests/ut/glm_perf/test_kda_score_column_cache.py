# SPDX-License-Identifier: Apache-2.0
"""Run the actual staged score loop in a CPU AscendC arithmetic simulation.

These tests cover output preservation, UB bounds, layouts and copy readiness.
Native code generation, device events and serving speed still need NPU gates.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "artifacts/glm-perf-310p/prompt-profile-20261005"


def compiler():
    result = shutil.which("c++")
    if result is None:
        pytest.skip("requires host C++ compiler")
    return result


def method(source, name):
    start = source.index("    __aicore__ inline ")
    start = source.index(name + "(", start)
    start = source.rfind("    __aicore__", 0, start)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture
def sources(tmp_path):
    metadata = json.loads((STUDY / "kda-score-cache-base.json").read_text())
    installed = (ROOT / metadata["path"]).read_text()
    target = tmp_path / metadata["path"]
    target.parent.mkdir(parents=True)
    target.write_text(installed)
    if hashlib.sha256(installed.encode()).hexdigest() == metadata["candidate_sha256"]:
        subprocess.run(
            ["git", "apply", "--reverse", str(STUDY / "kda-score-cache-columns.patch")], cwd=tmp_path, check=True
        )
    original = target.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == metadata["source_sha256"]
    subprocess.run(["git", "apply", str(STUDY / "kda-score-cache-columns.patch")], cwd=tmp_path, check=True)
    candidate = target.read_text()
    assert hashlib.sha256(candidate.encode()).hexdigest() == metadata["candidate_sha256"]
    return original, candidate


def test_disabled_flag_preserves_score_method(sources, tmp_path):
    outputs = []
    for source in sources:
        path = tmp_path / "method.cpp"
        path.write_text(method(source, "ComputeRawAqkAkkVector310P"))
        result = subprocess.run([compiler(), "-E", "-P", str(path)], check=True, capture_output=True, text=True)
        outputs.append(result.stdout.split())
    assert outputs[0] == outputs[1]


def test_actual_score_loop_preserves_outputs_and_reduces_reads(sources, tmp_path):
    prelude = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>
#define __aicore__
constexpr int PIPE_V=0;
constexpr unsigned EXP2_EVENT_ID=0;
constexpr unsigned KDA_VEC_ARENA_ELEMENTS=32768;
constexpr float LN2=0.69314718055994530942f;
enum class HardEvent {MTE2_V,V_S,S_V,V_MTE3,MTE3_MTE2,MTE3_V};
enum class RoundMode {CAST_NONE};
std::vector<std::function<void()>> pendingCopies;
template<int> void PipeBarrier() {}
template<HardEvent> void SetFlag(unsigned) {}
template<HardEvent E> void WaitFlag(unsigned) {
    if constexpr(E==HardEvent::MTE2_V) {
        for(auto& copy:pendingCopies) copy();
        pendingCopies.clear();
    }
}
template<class T> struct LocalTensor {
    T* pointer; uint8_t* end;
    LocalTensor operator[](uint64_t offset) const {
        assert(reinterpret_cast<uint8_t*>(pointer+offset)<=end);
        return {pointer+offset,end};
    }
    T GetValue(uint64_t i) const {
        assert(reinterpret_cast<uint8_t*>(pointer+i+1)<=end);
        return pointer[i];
    }
    void SetValue(uint64_t i,T x) { operator[](i).GetValue(0);pointer[i]=x; }
};
template<class T> using GlobalTensor=std::vector<T>;
struct Buffer {
    std::vector<uint64_t> storage=std::vector<uint64_t>(KDA_VEC_ARENA_ELEMENTS*4/8,0xA5A5A5A5A5A5A5A5);
    template<class T> LocalTensor<T> Get() {
        return {reinterpret_cast<T*>(storage.data()),
                reinterpret_cast<uint8_t*>(storage.data()+storage.size())};
    }
};
template<class T> void Duplicate(LocalTensor<T> a,T value,unsigned n) {
    for(unsigned i=0;i<n;++i)a.SetValue(i,value);
}
template<class T> void Sub(LocalTensor<T>a,LocalTensor<T>b,LocalTensor<T>c,unsigned n) {
    for(unsigned i=0;i<n;++i)a.SetValue(i,T(b.GetValue(i)-c.GetValue(i)));
}
template<class T> void Mul(LocalTensor<T>a,LocalTensor<T>b,LocalTensor<T>c,unsigned n) {
    for(unsigned i=0;i<n;++i)a.SetValue(i,T(b.GetValue(i)*c.GetValue(i)));
}
template<class T> void Muls(LocalTensor<T>a,LocalTensor<T>b,T c,unsigned n) {
    for(unsigned i=0;i<n;++i)a.SetValue(i,T(b.GetValue(i)*c));
}
template<class T> void Exp(LocalTensor<T>a,LocalTensor<T>b,unsigned n) {
    for(unsigned i=0;i<n;++i)a.SetValue(i,T(std::exp(float(b.GetValue(i)))));
}
template<class T> void Cast(LocalTensor<float>a,LocalTensor<T>b,RoundMode,unsigned n) {
    for(unsigned i=0;i<n;++i)a.SetValue(i,float(b.GetValue(i)));
}
void WholeReduceSum(LocalTensor<float>a,LocalTensor<float>b,unsigned width,int,int,int,int) {
    std::vector<float> row(width);
    for(unsigned i=0;i<width;++i)row[i]=b.GetValue(i);
    for(unsigned size=width;size>1;) {
        unsigned half=(size+1)/2;
        for(unsigned i=0;i<size/2;++i)row[i]=row[i]+row[i+half];
        size=half;
    }
    a.SetValue(0,row[0]);
}
void DataCopy(float& dst,LocalTensor<float>src,unsigned n) {
    src.GetValue(n-1); std::memcpy(&dst,src.pointer,n*4);
}
"""
    members = r"""
    uint64_t T_=160,H_=3,HV_=6,K_=128,BT_=64;
    bool inputSequenceMajor_=false;
    unsigned mte2ToVEvent_=0,vToMte3Event_=1,mte3ToVEvent_=2;
    unsigned mte3ToMte2Events_[1]={3};
    unsigned gmCopies=0; uint64_t gmBytes=0;
    Buffer vecBuf_;
    GlobalTensor<T> q_,k_,gk_;
    GlobalTensor<float> aqk_,akk_;
    void CopyVectorIn(LocalTensor<T>& dst,GlobalTensor<T>& src,uint64_t offset,uint64_t count) {
        assert(offset+count<=src.size());dst.GetValue(count-1);++gmCopies;gmBytes+=count*sizeof(T);
        pendingCopies.push_back([dst,&src,offset,count]() {
            std::memcpy(dst.pointer,src.data()+offset,count*sizeof(T));
        });
    }
    void ClampFp16ExpInput(LocalTensor<T>& x,unsigned count) {
        for(unsigned i=0;i<count;++i) {
            float v=float(x.GetValue(i));
            if(!std::isnan(v))v=std::clamp(v,-10.0f,10.0f);
            x.SetValue(i,T(v));
        }
    }
    void Setup(unsigned channels,unsigned blockSize,bool sequenceMajor,unsigned seed,bool exceptional) {
        K_=channels;BT_=blockSize;inputSequenceMajor_=sequenceMajor;
        q_.resize(2*H_*T_*K_);k_.resize(q_.size());gk_.resize(2*HV_*T_*K_);
        aqk_.assign(2*HV_*T_*BT_,-987.0f);akk_=aqk_;
        for(unsigned i=0;i<q_.size();++i) {
            q_[i]=T(std::sin(float(i+seed))*0.15f);
            k_[i]=T(std::cos(float(i+seed))*0.2f);
        }
        for(unsigned i=0;i<gk_.size();++i)gk_[i]=T(float((i+seed)%257)*-0.015f);
        if(exceptional) {
            for(unsigned i=0;i<q_.size();i+=97)q_[i]=T(-0.0f);
            for(unsigned i=11;i<k_.size();i+=337)k_[i]=T(std::numeric_limits<float>::quiet_NaN());
        }
    }
"""
    classes = []
    for name, source in zip(("Baseline", "Candidate"), sources):
        methods = "\n".join(
            method(source, member)
            for member in ("QOffset", "KVOffset", "AOffset", "ReduceDotProduct310P", "ComputeRawAqkAkkVector310P")
        )
        if name == "Candidate":
            methods = "#define GLM_KDA_SCORE_CACHE_COLUMNS\n" + methods + "\n#undef GLM_KDA_SCORE_CACHE_COLUMNS"
        classes.append(f"template<class T> struct {name} {{\n" + members + methods + "\n};\n")
    main = r"""
unsigned cases=0;
void Check(unsigned rows,unsigned channels,unsigned blockSize,bool sequenceMajor,unsigned seed,bool exceptional) {
    Baseline<_Float16> baseline;Candidate<_Float16> candidate;
    baseline.Setup(channels,blockSize,sequenceMajor,seed,exceptional);
    candidate.Setup(channels,blockSize,sequenceMajor,seed,exceptional);
    const auto q=candidate.q_,k=candidate.k_,g=candidate.gk_;
    for(unsigned head=0;head<3;++head) {
        baseline.ComputeRawAqkAkkVector310P(1,head,head+3,7,rows);
        candidate.ComputeRawAqkAkkVector310P(1,head,head+3,7,rows);
    }
    assert(pendingCopies.empty());
    assert(std::memcmp(baseline.aqk_.data(),candidate.aqk_.data(),baseline.aqk_.size()*4)==0);
    assert(std::memcmp(baseline.akk_.data(),candidate.akk_.data(),baseline.akk_.size()*4)==0);
    assert(std::memcmp(q.data(),candidate.q_.data(),q.size()*2)==0);
    assert(std::memcmp(k.data(),candidate.k_.data(),k.size()*2)==0);
    assert(std::memcmp(g.data(),candidate.gk_.data(),g.size()*2)==0);
    unsigned expectedBase=3*(4*rows+2*rows*(rows+1));
    bool cache=blockSize==64&&rows==64&&channels>=16&&channels<=256&&channels%16==0&&
               80*1024+2*rows*channels*sizeof(_Float16)<=KDA_VEC_ARENA_ELEMENTS*sizeof(float);
    unsigned expectedCandidate=cache?3*(4*rows+(sequenceMajor?rows:1)+1):expectedBase;
    assert(baseline.gmCopies==expectedBase);assert(candidate.gmCopies==expectedCandidate);
    assert(baseline.gmBytes==expectedBase*channels*2);
    assert(candidate.gmBytes==(cache?3*6*rows:expectedBase)*channels*2);
    // Score scratch ends below 16 KiB. Preserve the later triangular solve's
    // arena, including its high reduction scratch, across score computation.
    assert(std::memcmp(reinterpret_cast<uint8_t*>(baseline.vecBuf_.storage.data())+16*1024,
                       reinterpret_cast<uint8_t*>(candidate.vecBuf_.storage.data())+16*1024,
                       64*1024)==0);
    if(rows==64&&channels==128&&!sequenceMajor&&!exceptional)
        std::cout<<"64x128 copies: "<<baseline.gmCopies/3<<" -> "<<candidate.gmCopies/3<<"\n";
    ++cases;
}
int main() {
    for(bool layout:{false,true}) {
        for(unsigned rows:{0u,1u,2u,7u,16u,31u,63u,64u})Check(rows,128,64,layout,19,false);
        for(unsigned width:{16u,64u,192u,256u})Check(64,width,64,layout,71,false);
        for(unsigned rows:{47u,48u,49u})Check(rows,256,64,layout,23,false);
        Check(65,128,128,layout,7,false);Check(11,128,128,layout,43,false);
        Check(31,128,64,layout,97,true);Check(64,128,64,layout,5,true);
    }
    std::cout<<"cases="<<cases<<"\n";
}
"""
    path = tmp_path / "simulate.cpp"
    path.write_text(prelude + "\n".join(classes) + main)
    binary = tmp_path / "simulate"
    subprocess.run(
        [compiler(), "-std=c++17", "-O2", "-ffp-contract=off", "-fno-strict-aliasing", str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "copies: 8576 -> 258" in result.stdout
    assert "cases=38" in result.stdout
