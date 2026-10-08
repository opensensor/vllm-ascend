# SPDX-License-Identifier: Apache-2.0
"""Batch independent QSA softmax rows; preserve each row's 64-lane reduction."""

FLAG = "GLM_QSA_SOFTMAX_HEAD_BATCH"
EXTRA_UB_BYTES = 4 * 16 * 4

BATCH = """#ifdef GLM_QSA_SOFTMAX_HEAD_BATCH
        // Event 7 is unused by the qualified QSA parent (events 0 through 6).
        constexpr int64_t FP32_REPEAT_ELEMENTS = 64;
        constexpr int64_t FP32_REPEAT_BLOCKS = 8;
        auto maxima = softmaxBatchBuf_.Get<float>();
        auto sums = maxima[2 * MAX_QUERY_HEADS];
        auto deltas = maxima[3 * MAX_QUERY_HEADS];
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int64_t head = 0; head < queryHeads; ++head)
            for (int64_t lane = tileTokens; lane < tileWidth; ++lane)
                softmaxRows.SetValue(head * TOKEN_TILE + lane, -3.402823466e38f);
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        // ORDER_VALUE_INDEX has a two-float destination repeat; sum has one.
        WholeReduceMax(maxima, softmaxRows, FP32_REPEAT_ELEMENTS, queryHeads, 1, 1,
                       FP32_REPEAT_BLOCKS, ReduceOrder::ORDER_VALUE_INDEX);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int64_t head = 0; head < MAX_QUERY_HEADS; ++head)
            deltas.SetValue(head, 0.0f);
        for (int64_t head = 0; head < queryHeads; ++head) {
            const float tileMax = maxima.GetValue(2 * head);
            const float newMax = rowSum[head] == 0.0f || tileMax > rowMax[head] ? tileMax : rowMax[head];
            deltas.SetValue(head, rowSum[head] == 0.0f ? 0.0f : rowMax[head] - newMax);
            Adds(softmaxRows[head * TOKEN_TILE], softmaxRows[head * TOKEN_TILE], -newMax, tileWidth);
            rowMax[head] = newMax;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        Exp(deltas, deltas, MAX_QUERY_HEADS);
        PipeBarrier<PIPE_V>();
        Exp(softmaxRows, softmaxRows, queryHeads * tileWidth);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int64_t head = 0; head < queryHeads; ++head) {
            previousWeight[head] = rowSum[head] == 0.0f ? 0.0f : deltas.GetValue(head);
            for (int64_t lane = tileTokens; lane < tileWidth; ++lane)
                softmaxRows.SetValue(head * TOKEN_TILE + lane, 0.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        WholeReduceSum(sums, softmaxRows, FP32_REPEAT_ELEMENTS, queryHeads, 1, 1, FP32_REPEAT_BLOCKS);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int64_t head = 0; head < queryHeads; ++head)
            rowSum[head] = rowSum[head] * previousWeight[head] + sums.GetValue(head);
#else
"""


def transform(source):
    if FLAG in source:
        raise ValueError("QSA softmax head batch already staged")
    if "EVENT_ID7" in source:
        raise ValueError("qualified QSA event 7 is already in use")
    start = (
        "        for (int64_t head = 0; head < queryHeads; ++head) {\n"
        "            const int64_t rowOffset = head * TOKEN_TILE;"
    )
    end = "        Cast(probabilityRows, softmaxRows, RoundMode::CAST_NONE, queryHeads * tileWidth);"
    init = "        pipe_->InitBuffer(reduceBuf_, 2 * INT32_ALIGNMENT * sizeof(float));"
    member = "    TBuf<TPosition::VECCALC> reduceBuf_;"
    if any(source.count(anchor) != count for anchor, count in ((start, 2), (end, 1), (init, 1), (member, 1))):
        raise ValueError("qualified QSA softmax anchors changed")
    begin = source.index(start)
    boundary = source.index(end, begin)
    original = source[begin:boundary]
    if any(original.count(op) != 1 for op in ("ReduceMaximum(", "ReduceSum(", "ScalarExp(", "        Exp(")):
        raise ValueError("qualified QSA softmax arithmetic changed")
    source = source[:begin] + BATCH + original + "#endif\n" + source[boundary:]
    source = source.replace(
        init,
        init
        + "\n#ifdef "
        + FLAG
        + "\n        pipe_->InitBuffer(softmaxBatchBuf_, 4 * MAX_QUERY_HEADS * sizeof(float));\n#endif",
    )
    return source.replace(
        member, member + "\n#ifdef " + FLAG + "\n    TBuf<TPosition::VECCALC> softmaxBatchBuf_;\n#endif"
    )
