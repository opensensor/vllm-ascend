# SPDX-License-Identifier: Apache-2.0
"""Generate an isolated full KDA row-batching patch over qualified column caching."""

import difflib
import hashlib
import json
import re
from pathlib import Path


def main():
    study = Path(__file__).resolve().parent
    root = study.parents[2]
    relative = "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"
    baseline = (root / relative).read_text()
    expected = "ac922b4b01240a51757f0cf0ee6fa0d1b6937373e346e12318deb07945f36740"
    assert hashlib.sha256(baseline.encode()).hexdigest() == expected
    probe = (study / "kda-score-probe.cpp").read_text()
    start = probe.index("    __aicore__ inline void Compute(unsigned head)")
    opening = probe.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (probe[end] == "{") - (probe[end] == "}")
        end += 1
    method = probe[start:end]
    method = method.replace(
        "Compute(unsigned head)", "ComputeRawAqkAkkBatched310P(uint64_t b, uint64_t h, uint64_t hv, uint64_t start)"
    )
    method = method.replace(
        "auto halfArena = arena_.Get<T>();",
        "constexpr unsigned K = 128;\n        constexpr unsigned ROWS = 64;\n"
        "        constexpr unsigned BANK = K * ROWS;\n        auto halfArena = scoreBatchBuf_.Get<T>();",
    )
    method = method.replace("arena_.Get<float>()", "scoreBatchBuf_.Get<float>()")
    method = method.replace(
        "DataCopy(keys, k_[head * BANK], BANK);",
        """if (inputSequenceMajor_) {
            for (uint64_t col = 0; col < ROWS; ++col) {
                auto column = keys[col * K];
                CopyVectorIn(column, k_, QOffset(b, h, start + col, 0), K);
            }
        } else {
            CopyVectorIn(keys, k_, QOffset(b, h, start, 0), BANK);
        }""",
    )
    method = method.replace(
        "DataCopy(gates, g_[head * BANK], BANK);", "CopyVectorIn(gates, gk_, KVOffset(b, hv, start, 0, K), BANK);"
    )
    for source, tensor, offset in (
        ("rowVector, k_[head * BANK + row * K]", "k_", "QOffset(b, h, start + row, 0)"),
        ("rowVector, q_[head * BANK + row * K]", "q_", "QOffset(b, h, start + row, 0)"),
        ("gateRow, g_[head * BANK + row * K]", "gk_", "KVOffset(b, hv, start + row, 0, K)"),
    ):
        destination = source.split(",")[0]
        method = method.replace(f"DataCopy({source}, K);", f"CopyVectorIn({destination}, {tensor}, {offset}, K);")
    method = method.replace("[head * ROWS * ROWS + row * ROWS]", "[AOffset(b, hv, start + row, 0)]")
    for kind, event in (
        ("MTE2_V", "mte2ToVEvent_"),
        ("V_MTE3", "vToMte3Event_"),
        ("MTE3_MTE2", "mte3ToMte2Events_[0]"),
        ("MTE3_V", "mte3ToVEvent_"),
        ("V_S", "EXP2_EVENT_ID"),
        ("S_V", "EXP2_EVENT_ID"),
    ):
        for action in ("SetFlag", "WaitFlag"):
            method = re.sub(rf"{action}<HardEvent::{kind}>\(\d\)", f"{action}<HardEvent::{kind}>({event})", method)
    allocation = """
#if defined(GLM_KDA_SCORE_BATCH_ROWS) && defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
            if constexpr (SAFE_GATE && IsSameType<T, half>::value) {
                constexpr uint32_t scoreFinalizeStage = 7;  // KdaForward host stage contract.
                if (tiling.stage == scoreFinalizeStage && BT_ == 64 && K_ == 128) {
                    // Original 128 KiB solve arena + 16 KiB writeback + 1.5 KiB
                    // exp scratch + this 104 KiB bank fit within the 256 KiB UB.
                    pipe_->InitBuffer(scoreBatchBuf_, 104 * 1024);
                }
            }
#endif
"""
    candidate = baseline.replace("            AllocVectorEvents();", allocation + "            AllocVectorEvents();", 1)
    candidate = candidate.replace(
        "    TBuf<TPosition::VECCALC> gateWritebackBuf_;",
        """    TBuf<TPosition::VECCALC> gateWritebackBuf_;
#ifdef GLM_KDA_SCORE_BATCH_ROWS
    TBuf<TPosition::VECCALC> scoreBatchBuf_;
#endif""",
        1,
    )
    anchor = "    __aicore__ inline void ReduceDotProduct310P("
    candidate = candidate.replace(anchor, "#ifdef GLM_KDA_SCORE_BATCH_ROWS\n" + method + "\n#endif\n\n" + anchor, 1)
    anchor = """        // dav-m200 Cube does not commit partial M/N score tiles. Safe-gate"""
    dispatch = """#ifdef GLM_KDA_SCORE_BATCH_ROWS
        if constexpr (SAFE_GATE && IsSameType<T, half>::value) {
            if (BT_ == 64 && K_ == 128 && curT == 64) {
                ComputeRawAqkAkkBatched310P(b, h, hv, start);
                return;
            }
        }
#endif
"""
    candidate = candidate.replace(anchor, dispatch + anchor, 1)
    assert candidate != baseline
    (study / "kda-score-batch.patch").write_text(
        "".join(
            difflib.unified_diff(
                baseline.splitlines(keepends=True),
                candidate.splitlines(keepends=True),
                fromfile="a/" + relative,
                tofile="b/" + relative,
            )
        )
    )
    (study / "batch-header.h").write_text(candidate)
    metadata = {
        "path": relative,
        "source_sha256": expected,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "compile_flag": "GLM_KDA_SCORE_BATCH_ROWS",
        "arena_bytes": 104 * 1024,
        "dispatch": "310P FP16 safe-gate, full 64x128 chunks only; both public layouts",
    }
    (study / "kda-score-batch-base.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
