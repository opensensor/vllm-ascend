# SPDX-License-Identifier: Apache-2.0
"""Stage a separate KDA candidate that preserves both 64-lane score reductions."""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_KDA_SCORE_VECTOR_REDUCE"
EXTRA_UB_BYTES = 2 * 64 * 8 * 4 + 64 * 4
INITIALIZE = """#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && defined(GLM_KDA_SCORE_VECTOR_REDUCE)
            pipe_->InitBuffer(scoreReduceFirstBuf_, 64 * 8 * sizeof(float));
            pipe_->InitBuffer(scoreReduceSecondBuf_, 64 * 8 * sizeof(float));
            pipe_->InitBuffer(scoreReduceIndicesBuf_, 64 * sizeof(uint32_t));
#endif
"""
MEMBERS = """#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && defined(GLM_KDA_SCORE_VECTOR_REDUCE)
    TBuf<TPosition::VECCALC> scoreReduceFirstBuf_;
    TBuf<TPosition::VECCALC> scoreReduceSecondBuf_;
    TBuf<TPosition::VECCALC> scoreReduceIndicesBuf_;
#endif
"""
HELPER = """#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
    __aicore__ inline void ReduceScoreToVector310P(
        uint64_t col, LocalTensor<float> product)
    {
        constexpr uint32_t reduceWidth = 64;
        constexpr uint32_t stride = 8;
        LocalTensor<float> first = scoreReduceFirstBuf_.Get<float>()[col * stride];
        LocalTensor<float> second = scoreReduceSecondBuf_.Get<float>()[col * stride];
        WholeReduceSum(first, product, reduceWidth, 1, 1, 1, stride);
        WholeReduceSum(second, product[reduceWidth], reduceWidth, 1, 1, 1, stride);
        PipeBarrier<PIPE_V>();
    }
#endif

"""
PREPARE = """#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
        // Per-dot V/S synchronization also orders uncached column DMA.
        // Remove it only when the qualified full-chunk column cache is live.
#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
        const bool vectorScoreReduce = SAFE_GATE && IsSameType<T, half>::value &&
            K_ == 128 && BT_ == 64 && curT == 64 && cacheColumns;
#else
        const bool vectorScoreReduce = false;
#endif
        LocalTensor<float> scoreFirst = scoreReduceFirstBuf_.Get<float>();
        LocalTensor<float> scoreSecond = scoreReduceSecondBuf_.Get<float>();
        LocalTensor<uint32_t> scoreIndices = scoreReduceIndicesBuf_.Get<uint32_t>();
        if (vectorScoreReduce) {
            for (uint32_t index = 0; index < 64; ++index) {
                scoreIndices.SetValue(index, index * 8 * sizeof(float));
            }
            SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            Duplicate(scoreSecond, 0.0f, 64 * 8);
            PipeBarrier<PIPE_V>();
        }
#endif
"""
ROW_INITIALIZE = """#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
            if (vectorScoreReduce) {
                Duplicate(scoreFirst, 0.0f, 64 * 8);
                Duplicate(scoreSecond, 0.0f, 64 * 8);
                PipeBarrier<PIPE_V>();
            }
#endif
"""
REDUCE = """#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
                if (vectorScoreReduce) {
                    ReduceScoreToVector310P(col, productFp32);
                } else {
#endif
                ReduceDotProduct310P(
                    scoreRow, col, productFp32, partials);
#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
                }
#endif"""
GATHER = """#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
            if (vectorScoreReduce) {
                // Combine all columns once per row, retaining +0 + p0 + p1.
                Adds(scoreFirst, scoreFirst, 0.0f, 64 * 8);
                PipeBarrier<PIPE_V>();
                Add(scoreFirst, scoreFirst, scoreSecond, 64 * 8);
                PipeBarrier<PIPE_V>();
                Gather(scoreRow, scoreFirst, scoreIndices, static_cast<uint32_t>(0), 64);
                PipeBarrier<PIPE_V>();
            }
#endif
"""


def transform(source):
    if FLAG in source:
        raise ValueError("KDA score reduction candidate already staged")
    begin = source.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
    end = source.index("\n#endif", source.index("\n    }", begin))
    method = source[begin:end]
    replacements = (
        (
            "        // Run K*K and Q*K in separate passes.",
            PREPARE + "\n        // Run K*K and Q*K in separate passes.",
            1,
        ),
        ("            Duplicate(scoreRow, 0.0f,", ROW_INITIALIZE + "            Duplicate(scoreRow, 0.0f,", 2),
        (
            "                ReduceDotProduct310P(\n                    scoreRow, col, productFp32, partials);",
            REDUCE,
            2,
        ),
        (
            "            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);",
            GATHER + "            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);",
            2,
        ),
    )
    for before, after, expected in replacements:
        if method.count(before) != expected:
            raise ValueError("KDA score source anchor changed: " + before.strip())
        method = method.replace(before, after)
    source = source[:begin] + HELPER + method + source[end:]
    for before, after in (
        ("            AllocVectorEvents();", INITIALIZE + "            AllocVectorEvents();"),
        (
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;",
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;\n" + MEMBERS,
        ),
    ):
        if source.count(before) != 1:
            raise ValueError("KDA score source anchor changed: " + before.strip())
        source = source.replace(before, after)
    return source


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("KDA score source differs from its qualified parent")
    candidate = transform(original.decode()).encode()
    with destination.open("xb") as output:
        output.write(candidate)
    return dict(
        source=str(source),
        source_sha256=expected_sha256,
        destination=str(destination),
        candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        compiler_define=FLAG,
        extra_ub_bytes=EXTRA_UB_BYTES,
        full_kda_evaluated=False,
        serving_evaluated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.source, args.destination, args.expected_sha256)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
