# SPDX-License-Identifier: Apache-2.0
"""Stage a disabled-by-default full KDA candidate in a separate source tree."""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_KDA_VECTOR_BETA_SCALE"
MAX_CHANNELS = 256
EXTRA_UB_BYTES = MAX_CHANNELS * (2 + 4 + 2)

INITIALIZE = """#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && defined(GLM_KDA_VECTOR_BETA_SCALE)
            pipe_->InitBuffer(betaScaleInputBuf_, 256 * sizeof(half));
            pipe_->InitBuffer(betaScaleFloatBuf_, 256 * sizeof(float));
            pipe_->InitBuffer(betaScaleOutputBuf_, 256 * sizeof(half));
#endif
"""
MEMBERS = """#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && defined(GLM_KDA_VECTOR_BETA_SCALE)
    TBuf<TPosition::VECCALC> betaScaleInputBuf_;
    TBuf<TPosition::VECCALC> betaScaleFloatBuf_;
    TBuf<TPosition::VECCALC> betaScaleOutputBuf_;
#endif
"""
VECTOR_ROWS = """#ifdef GLM_KDA_VECTOR_BETA_SCALE
        // Preserve FP32 beta products and the terminal FP16 round. Separate
        // buffers avoid the mixed-view export problem on dav-m200.
        if constexpr (IsSameType<T, half>::value) {
            constexpr uint64_t maxChannels = 256;
            constexpr uint64_t dmaHalfElements = 16;
            if (dim >= dmaHalfElements && dim <= maxChannels && dim % dmaHalfElements == 0) {
                LocalTensor<half> inputLocal = betaScaleInputBuf_.Get<half>();
                LocalTensor<float> floatLocal = betaScaleFloatBuf_.Get<float>();
                LocalTensor<half> outputLocal = betaScaleOutputBuf_.Get<half>();
                for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
                    const uint64_t token = start + rowBegin + localRow;
                    const uint64_t dstOffset = KVOffset(b, hv, token, 0, dim);
                    const uint64_t srcOffset = sourceSequenceMajor
                        ? VInputOffset(b, hv, token, 0) : dstOffset;
                    const float betaScale = static_cast<float>(beta_.GetValue(BetaOffset(b, hv, token)));
                    SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
                    WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
                    CopyVectorIn(inputLocal, src, srcOffset, dim);
                    SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                    WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                    Cast(floatLocal, inputLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(dim));
                    PipeBarrier<PIPE_V>();
                    Muls(floatLocal, floatLocal, betaScale, static_cast<uint32_t>(dim));
                    PipeBarrier<PIPE_V>();
                    Cast(outputLocal, floatLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(dim));
                    SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                    WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                    CopyVectorOut(dst, dstOffset, outputLocal, dim);
                    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
                    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
                    SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
                    WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
                }
                return;
            }
        }
#endif
"""


def transform(source):
    if FLAG in source:
        raise ValueError("KDA beta candidate already staged")
    anchors = (
        ("            AllocVectorEvents();", INITIALIZE + "            AllocVectorEvents();"),
        (
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;",
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;\n" + MEMBERS,
        ),
        ("        (void)rowLocal;\n", "        (void)rowLocal;\n" + VECTOR_ROWS),
    )
    if "ScaleTailRowsByBeta310P" not in source or "FloatToType<T>(value * betaScale)" not in source:
        raise ValueError("authoritative KDA beta scalar method changed")
    for before, after in anchors:
        if source.count(before) != 1:
            raise ValueError("KDA beta source anchor changed: " + before.strip())
        source = source.replace(before, after)
    return source


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("KDA beta source differs from its qualified parent")
    candidate = transform(original.decode()).encode()
    # Refuse overwriting another staged source or any active package.
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
