# SPDX-License-Identifier: Apache-2.0
"""Batch full-row score vector work in an isolated qualified gated-key parent."""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_KDA_SCORE_ROW_BATCH"
GATE_BYTE_OFFSET = 16 * 1024
FLOAT_PRODUCT_BYTE_OFFSET = 32 * 1024
MAX_PRODUCT_END_BYTE = 64 * 1024
COLUMN_CACHE_BYTE_OFFSET = 80 * 1024
EXTRA_UB_BYTES = 0

PREPARE = """#ifdef GLM_KDA_SCORE_ROW_BATCH
            // Raw-score vectors occupy the low 8 KiB of vecBuf_. The
            // immutable column cache starts at 80 KiB. Reuse this phase's
            // otherwise dead middle arena; later solve stages run afterward.
            constexpr uint32_t gateBatchByteOffset = 16 * 1024;
            constexpr uint32_t productFloatByteOffset = 32 * 1024;
            LocalTensor<T> batchHalf = typedArena[gateBatchByteOffset / sizeof(T)];
            LocalTensor<float> batchFloat = vecBuf_.Get<float>()[productFloatByteOffset / sizeof(float)];
            const uint32_t batchElements = static_cast<uint32_t>((row + 1) * K_);
            for (uint64_t col = 0; col <= row; ++col) {
                Sub(batchHalf[col * K_], gRow, cachedGates[col * K_], static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(batchHalf, batchHalf, static_cast<T>(LN2), batchElements);
            PipeBarrier<PIPE_V>();
            ClampFp16ExpInput(batchHalf, batchElements);
            Exp(batchHalf, batchHalf, batchElements);
            PipeBarrier<PIPE_V>();
            Mul(gatedKeys, cachedKeys, batchHalf, batchElements);
            PipeBarrier<PIPE_V>();
            // All gated keys now own separate storage. Reuse batchHalf for
            // rounded FP16 products in each of the two independent passes.
#endif
"""
REDUCE = """#ifdef GLM_KDA_SCORE_ROW_BATCH
            PipeBarrier<PIPE_V>();
            Cast(batchFloat, batchHalf, RoundMode::CAST_NONE, batchElements);
            PipeBarrier<PIPE_V>();
            for (uint64_t col = 0; col <= row; ++col) {
                LocalTensor<float> columnProduct = batchFloat[col * K_];
#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
                if (vectorScoreReduce) {
                    ReduceScoreToVector310P(col, columnProduct);
                } else {
#endif
                    ReduceDotProduct310P(scoreRow, col, columnProduct, partials);
#ifdef GLM_KDA_SCORE_VECTOR_REDUCE
                }
#endif
            }
#endif
"""


def transform(source):
    if FLAG in source:
        raise ValueError("KDA score row batch already staged")
    for bound in (
        "constexpr uint32_t KDA_VEC_ARENA_ELEMENTS = 32768;",
        "constexpr uint64_t fp32ArenaOffset = 1024;",
        "constexpr uint64_t scoreCacheByteOffset = 80 * 1024;",
    ):
        if source.count(bound) != 1:
            raise ValueError("qualified score arena bounds changed")
    marker = "#if defined(GLM_KDA_GATED_KEY_REUSE) && defined(GLM_KDA_SCORE_CACHE_COLUMNS)"
    begin = source.index(marker)
    end = source.index("            return;", begin)
    branch = source[begin:end]
    loop = "            for (uint64_t col = 0; col <= row; ++col) {"
    positions = [index for index in range(len(branch)) if branch.startswith(loop, index)]
    if (
        len(positions) != 2
        or branch.count("curT == 64 && cacheColumns") != 1
        or branch.count("SAFE_GATE && IsSameType<T, half>::value && K_ == 128 && BT_ == 64") != 1
    ):
        raise ValueError("qualified full-row gated-key passes changed")
    passes = [branch[positions[0] : positions[1]], branch[positions[1] :]]
    # Keep the old per-column bodies under #else for exact disabled behavior.
    # In the active path, only arithmetic timing is batched. Preserve every
    # FP16 intermediate round and both 64-lane FP32 reduction trees.
    for index, body in enumerate(passes):
        start = (
            body.index("                Sub(gate, gRow, gCol,")
            if index == 0
            else body.index("                kGated = gatedKeys[col * K_];")
        )
        tail = body.index("\n            }", start)
        old = body[start:tail]
        batched = (
            "#ifdef GLM_KDA_SCORE_ROW_BATCH\n"
            "                Mul(batchHalf[col * K_], rowVector, gatedKeys[col * K_], "
            "static_cast<uint32_t>(K_));\n#else\n" + old + "\n#endif"
        )
        body = body[:start] + batched + body[tail:]
        gather = (
            "#ifdef GLM_KDA_SCORE_VECTOR_REDUCE\n"
            "            if (vectorScoreReduce) {\n                // Combine all columns"
        )
        if body.count(gather) != 1:
            raise ValueError("qualified score reduction boundary changed")
        passes[index] = body.replace(gather, REDUCE + gather)
    branch = branch[: positions[0]] + PREPARE + passes[0] + passes[1]
    return source[:begin] + branch + source[end:]


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("score row batch source differs from qualified parent")
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
        gate_byte_offset=GATE_BYTE_OFFSET,
        float_product_byte_offset=FLOAT_PRODUCT_BYTE_OFFSET,
        max_product_end_byte=MAX_PRODUCT_END_BYTE,
        column_cache_byte_offset=COLUMN_CACHE_BYTE_OFFSET,
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
