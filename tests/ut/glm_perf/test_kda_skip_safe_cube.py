# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the staged 310P KDA host scheduling change.

Extracts the real host lambda/loops and device score method. The mock executor
checks dependency wiring and failure propagation, not CANN execution ordering.
"""

import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "artifacts/glm-perf-310p/next-queue-20261005"


@pytest.fixture
def sources(tmp_path):
    metadata = json.loads((STUDY / "kda-skip-safe-cube-base.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        data = (ROOT / path).read_bytes()
        if path.endswith("chunk_kda_fwd_prepare.h"):
            # The current header contains subsequent scheduling work. Replay
            # this historical patch against its exact archived baseline.
            data = gzip.decompress((STUDY / "kda-skip-safe-cube-prepare-base.h.gz").read_bytes())
        assert hashlib.sha256(data).hexdigest() == digest, path
    original = (ROOT / metadata["path"]).read_text()
    target = tmp_path / metadata["path"]
    target.parent.mkdir(parents=True)
    target.write_text(original)
    subprocess.run(["git", "apply", str(STUDY / "kda-skip-safe-cube.patch")], cwd=tmp_path, check=True)
    return original, target.read_text()


def compiler():
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("requires host C++ compiler")
    return executable


def compile_run(tmp_path, source, defines=()):
    path = tmp_path / "probe.cpp"
    path.write_text(source)
    executable = tmp_path / "probe"
    subprocess.run(
        [compiler(), "-std=c++17", "-O0", *defines, str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)


def test_disabled_flag_preserves_host_tokens(sources, tmp_path):
    outputs = []
    for source in sources:
        # Only preprocess the host source: CANN headers are not installed here.
        path = tmp_path / "host.cpp"
        path.write_text(re.sub(r"^#include.*$", "", source, flags=re.MULTILINE))
        result = subprocess.run([compiler(), "-E", "-P", str(path)], check=True, capture_output=True, text=True)
        outputs.append(result.stdout.split())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_host_loops_and_dependency_lambda(sources, tmp_path, enabled):
    original, candidate = sources
    constants = "\n".join(re.findall(r"^constexpr int64_t KDA_STAGE_.*;$", original, flags=re.MULTILINE))
    start = candidate.index("    const aclTensor *stageDependency = nullptr;")
    end = candidate.index("    l0op::KdaCoreOutputs result{};", start)
    launch = candidate[start:end]
    first = candidate.index("        for (int64_t stage : {KDA_STAGE_GATE_PREPARE", end)
    first_end = candidate.index("        const int64_t scorePlaneElements", first)
    second = candidate.index("        for (int64_t stage : {KDA_STAGE_POST_WU", first_end)
    second_end = candidate.index("    } else if (splitStages)", second)
    assert "if (splitStages && IsAscend310P())" in candidate[end:first]
    assert "stageDependency = akkCompute;" in candidate[first_end:second]
    prelude = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>
struct aclTensor { int stage; };
struct Executor {
    int failAlloc=-1, failStage=-1, allocations=0;
    std::array<aclTensor,32> tokens{};
    std::vector<int> stages, dependencies;
};
namespace DataType { constexpr int DT_FLOAT=0; }
namespace l0op {
using KdaCoreOutputs=std::array<const aclTensor*,16>;
template<class... Args> KdaCoreOutputs KdaChunkForward(Args... args) {
    auto values=std::make_tuple(args...);
    auto executor=std::get<36>(values);
    auto dependency=std::get<10>(values);
    auto token=std::get<34>(values);
    int stage=std::get<35>(values);
    executor->stages.push_back(stage);
    executor->dependencies.push_back(dependency ? dependency->stage : -1);
    const_cast<aclTensor*>(token)->stage=stage;
    KdaCoreOutputs result{};
    result.fill(token);
    if(stage==executor->failStage) result[3]=nullptr;
    return result;
}
}
const aclTensor* AllocTensor(Executor* executor,int,int) {
    int index=executor->allocations++;
    return index==executor->failAlloc ? nullptr : &executor->tokens.at(index);
}
#define CHECK_RET(condition,error) if(!(condition)) return error
constexpr int ACLNN_ERR_INNER_NULLPTR=-7;
enum class KdaFwdLayout {BSND};
struct Params {
    const aclTensor *aLogOptional=nullptr,*dtBiasOptional=nullptr;
    const aclTensor *cuSeqlensOptional=nullptr,*chunkIndicesOptional=nullptr;
    float scale=1,lowerBound=-5;
    int chunkSize=64;
    bool safeGate=true,useGateInKernel=true;
};
int run(Executor& executor,bool safe) {
    auto executorPtr=&executor;
    int placeholderShape=0;
    Params params;
    params.safeGate=safe;
    auto parsedLayout=KdaFwdLayout::BSND;
    aclTensor castResult{99};
    const aclTensor *qHead=nullptr,*kHead=nullptr,*vHead=nullptr,*gHead=nullptr,*betaHead=nullptr;
    const aclTensor *initialStateCompute=nullptr,*gkFp16Compute=nullptr,*betaFp16Compute=nullptr;
    const aclTensor *attnCompute=nullptr,*finalStateCompute=nullptr,*gkCompute=nullptr,*aqkCompute=nullptr;
    const aclTensor *akkCompute=&castResult,*wCompute=nullptr,*uCompute=nullptr,*qgCompute=nullptr;
    const aclTensor *kgCompute=nullptr,*vNewCompute=nullptr,*hCompute=nullptr,*qgScaledCompute=nullptr;
    const aclTensor *uSeedCompute=nullptr,*scoreScratchCompute=nullptr,*scoreMatricesCompute=nullptr;
"""
    body = launch + "l0op::KdaCoreOutputs result{};\n" + candidate[first:first_end]
    body += "stageDependency = akkCompute;\n" + candidate[second:second_end] + "return 0;\n}\n"
    main = r"""
int main() {
    for(bool safe : {false,true}) {
        bool skip=false;
#ifdef GLM_KDA_SKIP_SAFE_SCORE_CUBE
        skip=safe;
#endif
        std::vector<int> expected=skip ? std::vector<int>{0,7,8,1,4,2,3,5}
                                       : std::vector<int>{0,6,7,8,1,4,2,3,5};
        Executor executor;
        assert(run(executor,safe)==0);
        assert(executor.stages==expected);
        for(unsigned i=0;i<expected.size();++i) {
            int previous=i ? expected[i-1] : -1;
            if(expected[i]==1) previous=99; // Explicit cast/copy barrier remains.
            assert(executor.dependencies[i]==previous);
        }
        // Every allocation/kernel failure must stop before the next stage.
        for(unsigned i=0;i<expected.size();++i) {
            Executor allocationFailure;
            allocationFailure.failAlloc=i;
            assert(run(allocationFailure,safe)==ACLNN_ERR_INNER_NULLPTR);
            assert(allocationFailure.stages.size()==i);
            Executor kernelFailure;
            kernelFailure.failStage=expected[i];
            assert(run(kernelFailure,safe)==ACLNN_ERR_INNER_NULLPTR);
            assert(kernelFailure.stages.size()==i+1);
        }
    }
}
"""
    defines = ["-DGLM_KDA_SKIP_SAFE_SCORE_CUBE"] if enabled else []
    compile_run(tmp_path, prelude.replace("int run(", constants + "\nint run(") + body + main, defines)


def test_actual_score_method_is_noop_for_safe_gate(sources, tmp_path):
    # Hash-guarded with the host patch: changing the native implementation
    # requires re-auditing the premise for deleting its physical launch.
    source = (ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h").read_text()
    start = source.index("    __aicore__ inline void ComputeScores310P(")
    end = source.index("    __aicore__ inline void FinalizeScores310P(", start)
    method = source[start:end]
    prelude = r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#define __aicore__
template<bool SAFE_GATE> struct Probe {
    uint64_t BT_=64,K_=128,calls=0;
    uint64_t ScoreRefBlockSize() {return 32;}
    uint64_t ScoreRowBlockCount(uint64_t n,uint64_t row) {return n-row<32 ? n-row : 32;}
    template<class... Args> void ComputeRawAqkAkkCubeBlock(Args...) {++calls;}
"""
    main = r"""
};
int main() {
    for(uint64_t length : {0,1,32,63,64}) {
        for(uint64_t width : {8,16,128,256}) {
            Probe<true> safe; safe.K_=width;
            safe.ComputeScores310P(0,0,10,10+length,0);
            assert(safe.calls==0);
            Probe<false> other; other.K_=width;
            other.ComputeScores310P(0,0,10,10+length,0);
            assert(other.calls==((length==64 && width>=16) ? 2 : 0));
        }
    }
}
"""
    compile_run(tmp_path, prelude + method + main)
