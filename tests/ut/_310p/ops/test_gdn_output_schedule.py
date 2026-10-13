# SPDX-License-Identifier: Apache-2.0
"""Compile the actual scalar schedulers with a device-free SDK type shell."""

import shutil
import subprocess
from pathlib import Path

import pytest

SHELL = r"""
#include <cstdint>
#include <cstdio>
#define CATLASS_DEVICE
#define __gm__
#define GM_ADDR void*
#define unlikely(x) (x)
struct ChunkFwdOVllmTilingData {
    uint32_t shapeBatch, seqlen, kNumHead, vNumHead, kHeadDim,
             vHeadDim, chunkSize, isVariedLen, tokenBatch;
};
namespace AscendC {
    template<class T> struct GlobalTensor {
        T* data;
        void SetGlobalBuffer(T* p) { data = p; }
        T GetValue(uint32_t i) { return data[i]; }
    };
    uint32_t GetBlockIdx() { return 0; }
    uint32_t GetBlockNum() { return 8; }
    uint32_t GetSubBlockNum() { return 1; }
}
namespace Catlass {
    namespace Arch { struct CrossCoreFlag { int value; }; }
    namespace Gemm { struct GemmCoord { uint32_t m, n, k; }; }
}
#include "HEADER"
int main(int argc, char** argv) {
    const uint32_t hv = std::stoul(argv[1]);
    const uint32_t chunks = std::stoul(argv[2]);
    const uint32_t batch = std::stoul(argv[3]);
    const bool varlen = std::stoul(argv[4]);
    int64_t lengths[] = {0, 128, 192};
    int64_t indices[] = {0, 0, 0, 1, 1, 0};
    ChunkFwdOVllmTilingData tiling {varlen ? 1u : batch,
        varlen ? 192u : chunks * 64, hv / 3, hv, 128, 128, 64,
        uint32_t(varlen), varlen ? 2u : 1u};
    for (uint32_t core = 0; core < 8; core++) {
        Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube s;
        static_cast<Catlass::Gemm::Block::BlockSchedulerGdnFwdO&>(s).Init(lengths, indices, &tiling, core, 8);
        while (s.isRunning) {
            const auto previous = s.currStage;
            s.InitTask();
            if (!s.isRunning) {
                // The exhaustion iteration must select the preceding valid
                // lane for pipeline draining; no phantom head is scheduled.
                if (s.currStage == previous) return 2;
                break;
            }
            auto &o = s.GetCube1Offsets();
            std::printf("%u %u %u %u %u %u\n", s.shapeBatchIdx,
                s.chunkIdx, s.vHeadIdx, o.qkOffset, o.ovOffset, o.hOffset);
        }
    }
}
"""


@pytest.mark.parametrize("arch", ["arch20", "arch22"])
def test_actual_scheduler_covers_odd_head_boundaries_and_drains_tail(tmp_path, arch):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[4]
    header = root / f"csrc/moe/chunk_fwd_o_vllm/op_kernel/{arch}/gemm/block/block_scheduler_gdn_fwd_o.hpp"
    source = tmp_path / "schedule.cpp"
    source.write_text("#include <string>\n" + SHELL.replace("HEADER", str(header)))
    binary = tmp_path / "schedule"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, capture_output=True)
    for heads in (6, 9, 12):
        for chunks in (1, 2, 40):
            for batch in (1, 2, 3, 4, 6):
                output = subprocess.check_output([str(binary), str(heads), str(chunks), str(batch), "0"], text=True)
                rows = [tuple(map(int, line.split())) for line in output.splitlines()]
                expected = [
                    (
                        b,
                        c,
                        h,
                        ((b * (heads // 3) + h // 3) * chunks * 64 + c * 64) * 128,
                        ((b * heads + h) * chunks * 64 + c * 64) * 128,
                        ((b * heads + h) * chunks + c) * 128 * 128,
                    )
                    for b in range(batch)
                    for c in range(chunks)
                    for h in range(heads)
                ]
                assert sorted(rows) == sorted(expected)
        output = subprocess.check_output([str(binary), str(heads), "3", "1", "1"], text=True)
        rows = [tuple(map(int, line.split())) for line in output.splitlines()]
        assert len(rows) == len(set(rows)) == 3 * heads
        assert {(r[1], r[2]) for r in rows} == {(c, h) for c in range(3) for h in range(heads)}


def test_actual_state_scheduler_skips_untrained_lane_and_preserves_varlen_batches(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[4]
    header = root / "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch20/gemm/block/block_scheduler_gdn_fwd_h.hpp"
    shell = SHELL.split('#include "HEADER"')[0].replace(
        "T GetValue(uint32_t i) { return data[i]; }",
        "T GetValue(uint32_t i) { return data[i]; } void SetValue(uint32_t i, T v) { data[i] = v; }",
    )
    source = tmp_path / "state.cpp"
    source.write_text(
        "#include <string>\n"
        + shell
        + r"""
struct ChunkGatedDeltaRuleFwdHTilingData {
    uint32_t batch, seqlen, kNumHead, vNumHead, kHeadDim, vHeadDim,
        chunkSize, isVariedLen, shapeBatch, tokenBatch, useInitialState,
        storeFinalState, numSeqWorkspaceOffset, numChunksWorkspaceOffset;
};
#include "HEADER"
int main(int argc, char** argv) {
    uint32_t heads = std::stoul(argv[1]);
    uint32_t batch = std::stoul(argv[2]);
    uint32_t chunks = std::stoul(argv[3]);
    bool varlen = std::stoul(argv[4]);
    int64_t lengths[] = {0, 64, 192};
    int64_t workspace[64] = {};
    ChunkGatedDeltaRuleFwdHTilingData tiling {batch,
        varlen ? 192u : chunks * 64, heads / 3, heads, 128, 128, 64,
        uint32_t(varlen), varlen ? 1u : batch, varlen ? 2u : 1u, 1, 1, 0, 256};
    for (uint32_t core = 0; core < 8; core++) {
        Catlass::Gemm::Block::BlockSchedulerGdnFwdH s;
        s.Init(lengths, nullptr, &tiling, workspace, core, 8);
        while (s.isRunning) {
            s.InitTask();
            if (s.NeedProcessStage1()) {
                auto &o = s.GetStage1Offsets();
                if (!s.OwnsInitialState(o.batchIdx, o.headIdx)) return 3;
                for (uint32_t other = 0; other < 8; other++) {
                    if (other == core) continue;
                    auto original = s.cubeCoreIdx;
                    s.cubeCoreIdx = other;
                    if (s.OwnsInitialState(o.batchIdx, o.headIdx)) return 4;
                    s.cubeCoreIdx = original;
                }
                std::printf("%u %u %u %u %u\n", o.batchIdx, o.chunkIdx,
                    o.headIdx, o.initialStateOffset, o.hSrcOffset);
            }
        }
    }
}
""".replace("HEADER", str(header))
    )
    binary = tmp_path / "state"
    (tmp_path / "catlass").mkdir()
    (tmp_path / "catlass/gemm_coord.hpp").write_text("#pragma once\n")
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(tmp_path), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    for heads in (6, 9, 12):
        for batch in (1, 2, 3, 4, 6):
            for chunks in (1, 2, 40):
                text = subprocess.check_output([str(binary), str(heads), str(batch), str(chunks), "0"], text=True)
                rows = [tuple(map(int, line.split())) for line in text.splitlines()]
                expected = [
                    (b, c, h, (b * heads + h) * 128**2, ((b * heads + h) * chunks + c) * 128**2)
                    for b in range(batch)
                    for c in range(chunks)
                    for h in range(heads)
                ]
                assert sorted(rows) == sorted(expected)
        text = subprocess.check_output([str(binary), str(heads), "1", "3", "1"], text=True)
        rows = [tuple(map(int, line.split())) for line in text.splitlines()]
        expected = [
            (b, c, h, (b * heads + h) * 128**2, (h * 3 + b + c) * 128**2)
            for b, count in [(0, 1), (1, 2)]
            for c in range(count)
            for h in range(heads)
        ]
        assert sorted(rows) == sorted(expected)
