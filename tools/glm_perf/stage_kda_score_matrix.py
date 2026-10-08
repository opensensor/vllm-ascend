# SPDX-License-Identifier: Apache-2.0
"""Stage repeat-based KDA score arithmetic on the qualified full-row parent."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.glm_perf.stage_kda_score_row_batch import REDUCE

FLAG = "GLM_KDA_SCORE_MATRIX_BATCH"
SUB = """#ifdef GLM_KDA_SCORE_MATRIX_BATCH
            // K=128 half lanes occupy eight blocks. Repeat across independent
            // columns while keeping the row operand fixed (source stride zero).
            const BinaryRepeatParams matrixRepeat{1, 1, 1, 8, 0, 8};
            Sub(batchHalf, gRow, cachedGates, static_cast<uint64_t>(128), row + 1, matrixRepeat);
#else
"""
SUM = """#ifdef GLM_KDA_SCORE_MATRIX_BATCH
            PipeBarrier<PIPE_V>();
            Cast(batchFloat, batchHalf, RoundMode::CAST_NONE, batchElements);
            PipeBarrier<PIPE_V>();
            if (vectorScoreReduce) {
                // Keep two separate 64-lane trees and the parent's +0+p0+p1
                // combine. Sum destination stride is in float elements, not blocks.
                WholeReduceSum(scoreFirst, batchFloat, 64, row + 1, 8, 1, 16);
                WholeReduceSum(scoreSecond, batchFloat[64], 64, row + 1, 8, 1, 16);
                PipeBarrier<PIPE_V>();
            } else {
                for (uint64_t col = 0; col <= row; ++col) {
                    LocalTensor<float> columnProduct = batchFloat[col * K_];
                    ReduceDotProduct310P(scoreRow, col, columnProduct, partials);
                }
            }
#else
"""


def transform(source):
    if FLAG in source:
        raise ValueError("KDA score matrix already staged")
    sub = (
        "            for (uint64_t col = 0; col <= row; ++col) {\n"
        "                Sub(batchHalf[col * K_], gRow, cachedGates[col * K_], static_cast<uint32_t>(K_));\n"
        "            }\n"
    )
    product = (
        "            for (uint64_t col = 0; col <= row; ++col) {\n"
        "#ifdef GLM_KDA_SCORE_CACHE_COLUMNS\n"
        "                if (cacheColumns) {"
    )
    marker = "#if defined(GLM_KDA_GATED_KEY_REUSE) && defined(GLM_KDA_SCORE_CACHE_COLUMNS)"
    if source.count(marker) != 1:
        raise ValueError("qualified full-row score matrix branch changed")
    begin = source.index(marker)
    end = source.index("            return;", begin)
    prefix, suffix = source[:begin], source[end:]
    source = source[begin:end]
    guard = "SAFE_GATE && IsSameType<T, half>::value && K_ == 128 && BT_ == 64 && curT == 64 && cacheColumns"
    if source.count(sub) != 1 or source.count(product) != 2 or source.count(REDUCE) != 2 or source.count(guard) != 1:
        raise ValueError("qualified full-row score matrix anchors changed")
    if "scoreReduceFirstBuf_" not in prefix or "scoreCacheByteOffset = 80 * 1024;" not in prefix:
        raise ValueError("qualified score scratch changed")
    source = source.replace(sub, SUB + sub + "#endif\n")
    # Replace from right to left so searches cannot rediscover an inserted #else.
    starts = [index for index in range(len(source)) if source.startswith(product, index)]
    for begin in reversed(starts):
        end = source.index("\n            }", begin) + len("\n            }")
        original = source[begin:end]
        replacement = (
            "#ifdef " + FLAG + "\n"
            "            Mul(batchHalf, rowVector, gatedKeys, static_cast<uint64_t>(128), row + 1, matrixRepeat);\n"
            "#else\n" + original + "\n#endif"
        )
        source = source[:begin] + replacement + source[end:]
    source = source.replace(REDUCE, SUM + REDUCE + "#endif\n")
    requirement = """#if defined(GLM_KDA_SCORE_MATRIX_BATCH) && (!defined(GLM_KDA_SCORE_ROW_BATCH) || \\
    !defined(GLM_KDA_SCORE_VECTOR_REDUCE) || !defined(GLM_KDA_GATED_KEY_REUSE) || !defined(GLM_KDA_SCORE_CACHE_COLUMNS))
#error "score matrix requires the qualified row batch, reduction, gated-key and column-cache parent"
#endif
"""
    return requirement + prefix + source + suffix


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("score matrix source differs from qualified parent")
    candidate = transform(original.decode()).encode()
    with destination.open("xb") as output:
        output.write(candidate)
    return dict(
        source_sha256=expected_sha256,
        candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        compiler_define=FLAG,
        extra_ub_bytes=0,
        full_kda_evaluated=False,
        serving_evaluated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    report = stage(args.source, args.destination, args.expected_sha256)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
