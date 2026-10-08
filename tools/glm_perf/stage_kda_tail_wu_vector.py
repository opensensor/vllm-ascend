# SPDX-License-Identifier: Apache-2.0
"""Stage isolated vector tail dot products with the scalar FP32 sum order."""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_KDA_TAIL_WU_VECTOR"
EXTRA_UB_BYTES = 64 * (2 + 4) + 64 * 8 * 4 + 128 * (2 + 4 + 4 + 4 + 4 + 2 + 4)
INITIALIZE = """#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && defined(GLM_KDA_TAIL_WU_VECTOR)
            pipe_->InitBuffer(tailCoefficientsHalfBuf_, 64 * sizeof(half));
            pipe_->InitBuffer(tailCoefficientsFloatBuf_, 64 * sizeof(float));
            pipe_->InitBuffer(tailCoefficientsBroadcastBuf_, 64 * 8 * sizeof(float));
            pipe_->InitBuffer(tailValuesHalfBuf_, 128 * sizeof(half));
            pipe_->InitBuffer(tailValuesFloatBuf_, 128 * sizeof(float));
            pipe_->InitBuffer(tailBroadcastBuf_, 128 * sizeof(float));
            pipe_->InitBuffer(tailProductBuf_, 128 * sizeof(float));
            pipe_->InitBuffer(tailAccumulatorBuf_, 128 * sizeof(float));
            pipe_->InitBuffer(tailOutputBuf_, 128 * sizeof(half));
            pipe_->InitBuffer(tailIndicesBuf_, 128 * sizeof(uint32_t));
#endif
"""
MEMBERS = """#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && defined(GLM_KDA_TAIL_WU_VECTOR)
    TBuf<TPosition::VECCALC> tailCoefficientsHalfBuf_;
    TBuf<TPosition::VECCALC> tailCoefficientsFloatBuf_;
    TBuf<TPosition::VECCALC> tailCoefficientsBroadcastBuf_;
    TBuf<TPosition::VECCALC> tailValuesHalfBuf_;
    TBuf<TPosition::VECCALC> tailValuesFloatBuf_;
    TBuf<TPosition::VECCALC> tailBroadcastBuf_;
    TBuf<TPosition::VECCALC> tailProductBuf_;
    TBuf<TPosition::VECCALC> tailAccumulatorBuf_;
    TBuf<TPosition::VECCALC> tailOutputBuf_;
    TBuf<TPosition::VECCALC> tailIndicesBuf_;
#endif
"""
VECTOR = """#ifdef GLM_KDA_TAIL_WU_VECTOR
        if constexpr (IsSameType<T, half>::value && IsSameType<SrcTensor, half>::value &&
                      IsSameType<DstTensor, half>::value) {
            if (BT_ == 64 && curT > 0 && curT < 64 && dim == 128) {
                LocalTensor<half> coefficientHalf = tailCoefficientsHalfBuf_.Get<half>();
                LocalTensor<float> coefficientFloat = tailCoefficientsFloatBuf_.Get<float>();
                LocalTensor<float> coefficientBroadcast = tailCoefficientsBroadcastBuf_.Get<float>();
                LocalTensor<half> valueHalf = tailValuesHalfBuf_.Get<half>();
                LocalTensor<float> valueFloat = tailValuesFloatBuf_.Get<float>();
                LocalTensor<float> broadcast = tailBroadcastBuf_.Get<float>();
                LocalTensor<float> product = tailProductBuf_.Get<float>();
                LocalTensor<float> accumulator = tailAccumulatorBuf_.Get<float>();
                LocalTensor<half> output = tailOutputBuf_.Get<half>();
                LocalTensor<uint32_t> indices = tailIndicesBuf_.Get<uint32_t>();
                Duplicate(coefficientHalf, static_cast<half>(0.0f), 64);
                Duplicate(accumulator, 0.0f, static_cast<uint32_t>(dim));
                Duplicate(indices, static_cast<uint32_t>(0), static_cast<uint32_t>(dim));
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
                // Each prepared score row owns BT_=64 half elements even
                // for a partial chunk. Use aligned DMA of that complete row;
                // dav-m200 short DataCopyPad did not commit tail coefficients.
                CopyVectorIn(coefficientHalf, preparedAqk_, akkBase, 64);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                Cast(coefficientFloat, coefficientHalf, RoundMode::CAST_NONE, 64);
                PipeBarrier<PIPE_V>();
                Brcb(coefficientBroadcast, coefficientFloat, 8, {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint64_t j = 0; j < curT; ++j) {
                    CopyVectorIn(valueHalf, src, srcBase + j * rowStride, dim);
                    SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                    WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                    Cast(valueFloat, valueHalf, RoundMode::CAST_NONE, static_cast<uint32_t>(dim));
                    Gather(broadcast, coefficientBroadcast[j * 8], indices,
                           static_cast<uint32_t>(0), static_cast<uint32_t>(dim));
                    PipeBarrier<PIPE_V>();
                    // Products of two finite FP16 operands are exact in FP32.
                    // Keep the scalar reference's sequential +0 + p0 + p1 ...
                    // sum; never read a scalar into an accumulator's UB lane.
                    Mul(product, broadcast, valueFloat, static_cast<uint32_t>(dim));
                    PipeBarrier<PIPE_V>();
                    Add(accumulator, accumulator, product, static_cast<uint32_t>(dim));
                    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
                }
                Mins(accumulator, accumulator, KDA_FP16_MAX, static_cast<uint32_t>(dim));
                PipeBarrier<PIPE_V>();
                Maxs(accumulator, accumulator, -KDA_FP16_MAX, static_cast<uint32_t>(dim));
                PipeBarrier<PIPE_V>();
                Cast(output, accumulator, RoundMode::CAST_NONE, static_cast<uint32_t>(dim));
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                CopyVectorOut(dst, dstBase, output, dim);
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
                SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
                WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
                return;
            }
        }
#endif
"""


def transform(source):
    if FLAG in source:
        raise ValueError("KDA vector tail dots already staged")
    begin = source.index("    __aicore__ inline void ComputeTailWuRow(")
    end = source.index("    __aicore__ inline void ComputeTailWuVector(", begin)
    method = source[begin:end]
    anchor = "#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)\n"
    if method.count(anchor) != 1 or "acc += coefficient * value;" not in method:
        raise ValueError("authoritative scalar tail dot method changed")
    method = method.replace(anchor, anchor + VECTOR)
    source = source[:begin] + method + source[end:]
    for before, after in (
        ("            AllocVectorEvents();", INITIALIZE + "            AllocVectorEvents();"),
        (
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;",
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;\n" + MEMBERS,
        ),
    ):
        if source.count(before) != 1:
            raise ValueError("KDA vector tail source anchor changed: " + before.strip())
        source = source.replace(before, after)
    return source


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("KDA vector tail source differs from its qualified parent")
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
