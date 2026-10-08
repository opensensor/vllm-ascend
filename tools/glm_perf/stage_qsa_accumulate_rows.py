# SPDX-License-Identifier: Apache-2.0
"""Batch independent QSA accumulator lanes while retaining FP32 multiply/add order."""

FLAG = "GLM_QSA_ACCUMULATE_ROWS"


def transform(source):
    if FLAG in source:
        raise ValueError("QSA accumulator rows already staged")
    function = "    __aicore__ inline void AccumulateValues(int64_t queryHeads, const float *previousWeight)"
    boundary = "    __aicore__ inline void StoreOutput("
    if source.count(function) != 1 or source.count(boundary) != 1:
        raise ValueError("qualified QSA accumulator anchors changed")
    first = source.index(function)
    end = source.index(boundary, first)
    original = source[first:end]
    if any(
        original.count(op) != n for op, n in (("Muls(", 1), ("Add(", 1), ("Adds(", 1), ("PipeBarrier<PIPE_V>();", 2))
    ):
        raise ValueError("qualified QSA accumulation arithmetic changed")
    batched = """    __aicore__ inline void AccumulateValues(int64_t queryHeads, const float *previousWeight)
    {
        constexpr int64_t FP32_BLOCK = 8;
        constexpr uint8_t ROW_STRIDE = MAX_QUERY_HEADS * NZ_INNER / FP32_BLOCK;
        const UnaryRepeatParams unary{1, 1, ROW_STRIDE, ROW_STRIDE};
        const BinaryRepeatParams binary{1, 1, 1, ROW_STRIDE, ROW_STRIDE, ROW_STRIDE};
        LocalTensor<float> contribution = contributionBuf_.Get<float>();
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        for (int64_t head = 0; head < queryHeads; ++head) {
            const int64_t offset = head * NZ_INNER;
            if (previousWeight[head] == 0.0f) {
                Adds(accumulator[offset], contribution[offset], 0.0f, NZ_INNER, headDimBlocks_, unary);
            } else {
                Muls(accumulator[offset], accumulator[offset], previousWeight[head],
                     NZ_INNER, headDimBlocks_, unary);
                PipeBarrier<PIPE_V>();
                Add(accumulator[offset], accumulator[offset], contribution[offset],
                    NZ_INNER, headDimBlocks_, binary);
            }
        }
        PipeBarrier<PIPE_V>();
    }

"""
    return source[:first] + "#ifdef " + FLAG + "\n" + batched + "#else\n" + original + "#endif\n" + source[end:]
