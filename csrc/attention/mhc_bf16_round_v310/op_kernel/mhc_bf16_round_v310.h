#ifndef MHC_BF16_ROUND_V310_H
#define MHC_BF16_ROUND_V310_H

#include "kernel_operator.h"
#include "mhc_bf16_round_v310_tiling_data.h"

namespace NsMhcBf16Round {

using namespace AscendC;

constexpr int64_t TILE_ELEMENTS = 1024;

class MhcBf16RoundV310 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const MhcBf16RoundV310TilingData* tiling,
                                TPipe* pipe)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input), tiling->numel);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), tiling->numel);
        inputBitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(input), tiling->numel);
        outputBitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output), tiling->numel);
        numel_ = tiling->numel;
        blockCount_ = tiling->blockCount;
        pipe->InitBuffer(buffer_, TILE_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto localFloat = buffer_.Get<float>();
        auto localBits = localFloat.ReinterpretCast<uint32_t>();
        for (int64_t offset = GetBlockIdx() * TILE_ELEMENTS;
             offset < numel_; offset += blockCount_ * TILE_ELEMENTS) {
            const int64_t remaining = numel_ - offset;
            const int32_t elements = remaining < TILE_ELEMENTS ? remaining : TILE_ELEMENTS;
            const int32_t aligned = elements & ~7;
            if (aligned > 0) {
                DataCopy(localFloat, inputGm_[offset], aligned);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (int32_t index = 0; index < aligned; ++index) {
                    localBits.SetValue(index, RoundBits(localBits.GetValue(index)));
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(outputGm_[offset], localFloat, aligned);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
            // Scalar GM access avoids an unsupported unaligned DataCopyPad
            // tail on the DAV-M200 path; at most seven elements reach here.
            for (int32_t index = aligned; index < elements; ++index) {
                outputBitsGm_.SetValue(offset + index,
                                       RoundBits(inputBitsGm_.GetValue(offset + index)));
            }
            if (aligned > 0) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }

private:
    __aicore__ inline uint32_t RoundBits(uint32_t bits)
    {
        if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0) {
            // The 310P BF16 cast canonicalizes either NaN sign to a quiet NaN.
            return (bits & 0x80000000U) | 0x7fc00000U;
        }
        // Round to nearest even BF16 without an FP32->BF16 AI-CPU cast.
        return (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
    }

    GlobalTensor<float> inputGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<uint32_t> inputBitsGm_;
    GlobalTensor<uint32_t> outputBitsGm_;
    TBuf<TPosition::VECCALC> buffer_;
    int64_t numel_ = 0;
    int64_t blockCount_ = 0;
};

}  // namespace NsMhcBf16Round

#endif
