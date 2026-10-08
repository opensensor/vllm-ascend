# SPDX-License-Identifier: Apache-2.0
"""Batch QSA result scaling/casting across NZ blocks without changing arithmetic."""

FLAG = "GLM_QSA_OUTPUT_ROWS"


def transform(source):
    if FLAG in source:
        raise ValueError("QSA output rows already staged")
    start = (
        "        for (int64_t head = 0; head < queryHeads; ++head) {\n"
        "            const float inverse = rowSum[head] > 0.0f ? 1.0f / rowSum[head] : 0.0f;"
    )
    end = "        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("qualified QSA output loop changed")
    offset = source.index(start)
    boundary = source.index(end, offset)
    original = source[offset:boundary]
    if original.count("PipeBarrier<PIPE_V>();") != 1 or original.count("Cast(") != 1 or original.count("Muls(") != 1:
        raise ValueError("qualified QSA output arithmetic changed")
    batched = """        constexpr int64_t FP32_BLOCK = 8;
        const UnaryRepeatParams scaleStride{1, 1,
            static_cast<uint8_t>(MAX_QUERY_HEADS * NZ_INNER / FP32_BLOCK),
            static_cast<uint8_t>(MAX_QUERY_HEADS * NZ_INNER / FP32_BLOCK)};
        const UnaryRepeatParams castStride{1, 1, 1,
            static_cast<uint8_t>(MAX_QUERY_HEADS * NZ_INNER / FP32_BLOCK)};
        for (int64_t head = 0; head < queryHeads; ++head) {
            const float inverse = rowSum[head] > 0.0f ? 1.0f / rowSum[head] : 0.0f;
            Muls(accumulator[head * NZ_INNER], accumulator[head * NZ_INNER], inverse,
                 NZ_INNER, headDimBlocks_, scaleStride);
            PipeBarrier<PIPE_V>();
            Cast(output[head * headDim_], accumulator[head * NZ_INNER], RoundMode::CAST_NONE,
                 NZ_INNER, headDimBlocks_, castStride);
        }
"""
    return source[:offset] + "#ifdef " + FLAG + "\n" + batched + "#else\n" + original + "#endif\n" + source[boundary:]
