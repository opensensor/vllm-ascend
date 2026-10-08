# SPDX-License-Identifier: Apache-2.0
"""Export actual score methods for an isolated, device-free native build."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[3]
study = Path(__file__).resolve().parent
metadata = json.loads((study / "kda-score-cache-base.json").read_text())


def method(source, name):
    start = source.index(name + "(")
    start = source.rfind("    __aicore__", 0, start)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / metadata["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes((root / metadata["path"]).read_bytes())
    if hashlib.sha256(target.read_bytes()).hexdigest() == metadata["candidate_sha256"]:
        subprocess.run(
            ["git", "apply", "--reverse", str(study / "kda-score-cache-columns.patch")], cwd=directory, check=True
        )
    assert hashlib.sha256(target.read_bytes()).hexdigest() == metadata["source_sha256"]
    subprocess.run(["git", "apply", str(study / "kda-score-cache-columns.patch")], cwd=directory, check=True)
    source = target.read_text()

prefix = r"""// SPDX-License-Identifier: Apache-2.0
// Extracted scoring methods, not the complete KDA pipeline or serving binding.
#include "kernel_operator.h"
using namespace AscendC;
constexpr float LN2 = 0.69314718055994530942f;
constexpr float KDA_FP16_EXP_INPUT_MAX = 11.089866f;
constexpr float KDA_FP16_EXP_INPUT_MIN = -17.0f;
constexpr unsigned EXP2_EVENT_ID = 0;
constexpr unsigned KDA_VEC_ARENA_ELEMENTS = 32768;
template<class T> class ScoreProbe {
public:
    __aicore__ inline void Run(GM_ADDR q,GM_ADDR k,GM_ADDR g,GM_ADDR aqk,GM_ADDR akk) {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g));
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aqk));
        akk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(akk));
        pipe_.InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
        for(uint64_t head=GetBlockIdx();head<HV_;head+=GetBlockNum())
            ComputeRawAqkAkkVector310P(0,head,head,0,64);
    }
private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> vecBuf_;
    GlobalTensor<T> q_,k_,gk_;
    GlobalTensor<float> aqk_,akk_;
    uint64_t T_=64,H_=16,HV_=16,K_=128,BT_=64;
    bool inputSequenceMajor_=false;
    unsigned mte2ToVEvent_=0,vToMte3Event_=1,mte3ToVEvent_=2;
    unsigned mte3ToMte2Events_[1]={3};
    __aicore__ inline void CopyVectorIn(LocalTensor<T>& dst,GlobalTensor<T>& src,
                                      uint64_t offset,uint64_t count) {
        // Fixed 128-channel FP16 probe: every copy is 32-byte aligned.
        DataCopy(dst,src[offset],static_cast<uint32_t>(count));
    }
"""
methods = "\n".join(
    method(source, name)
    for name in (
        "QOffset",
        "KVOffset",
        "AOffset",
        "ClampFp16ExpInput",
        "ReduceDotProduct310P",
        "ComputeRawAqkAkkVector310P",
    )
)
suffix = r"""
};
extern "C" __global__ __aicore__ void glm_kda_score_probe_v1(
    GM_ADDR q,GM_ADDR k,GM_ADDR g,GM_ADDR aqk,GM_ADDR akk) {
    InitSocState();
    ScoreProbe<half> probe;
    probe.Run(q,k,g,aqk,akk);
}
"""
(study / "kda-score-probe.cpp").write_text(prefix + methods + suffix)
print("Exported", study / "kda-score-probe.cpp")
