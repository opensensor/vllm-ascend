// SPDX-License-Identifier: Apache-2.0
// Extracted scoring methods, not the complete KDA pipeline or serving binding.
#include "kernel_operator.h"
using namespace AscendC;
constexpr float LN2 = 0.69314718055994530942f;
constexpr float KDA_FP16_EXP_INPUT_MAX = 11.089866f;
constexpr float KDA_FP16_EXP_INPUT_MIN = -17.0f;
constexpr unsigned EXP2_EVENT_ID = 0;
constexpr unsigned KDA_VEC_ARENA_ELEMENTS = 32768;
template<class T> class ScoreProbe {
public:
    __aicore__ inline void Run(GM_ADDR q,GM_ADDR k,GM_ADDR g,GM_ADDR aqk,GM_ADDR akk) {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g));
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aqk));
        akk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(akk));
        pipe_.InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
        for(uint64_t head=GetBlockIdx();head<HV_;head+=GetBlockNum())
            ComputeRawAqkAkkVector310P(0,head,head,0,64);
    }
private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> vecBuf_;
    GlobalTensor<T> q_,k_,gk_;
    GlobalTensor<float> aqk_,akk_;
    uint64_t T_=64,H_=16,HV_=16,K_=128,BT_=64;
    bool inputSequenceMajor_=false;
    unsigned mte2ToVEvent_=0,vToMte3Event_=1,mte3ToVEvent_=2;
    unsigned mte3ToMte2Events_[1]={3};
    __aicore__ inline void CopyVectorIn(LocalTensor<T>& dst,GlobalTensor<T>& src,
                                      uint64_t offset,uint64_t count) {
        // Fixed 128-channel FP16 probe: every copy is 32-byte aligned.
        DataCopy(dst,src[offset],static_cast<uint32_t>(count));
    }
    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * H_ + h) * K_ + d;
        }
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }
    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }
    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t j) const
    {
        return ((b * HV_ + hv) * T_ + t) * BT_ + j;
    }
    __aicore__ inline void ClampFp16ExpInput(LocalTensor<T> &tensor,
                                              uint32_t count)
    {
        // exp(ln(65504)) is the largest finite FP16 exponential result.
        Mins(tensor, tensor, static_cast<T>(KDA_FP16_EXP_INPUT_MAX), count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, static_cast<T>(KDA_FP16_EXP_INPUT_MIN), count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ReduceDotProduct310P(
        LocalTensor<float> &dst, uint64_t dstOffset,
        LocalTensor<float> &product, LocalTensor<float> &partials)
    {
        constexpr uint64_t reduceWidth = 64;
        constexpr uint64_t resultStride = 8;
        uint64_t partialCount = 0;
        for (uint64_t offset = 0; offset < K_; offset += reduceWidth) {
            uint64_t width = K_ - offset;
            if (width > reduceWidth) {
                width = reduceWidth;
            }
            WholeReduceSum(
                partials[partialCount * resultStride], product[offset],
                static_cast<uint32_t>(width), 1, 1, 1, resultStride);
            ++partialCount;
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        float sum = 0.0f;
        for (uint64_t partial = 0; partial < partialCount; ++partial) {
            sum += partials.GetValue(partial * resultStride);
        }
        dst.SetValue(dstOffset, sum);
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
    }
    __aicore__ inline void ComputeRawAqkAkkVector310P(
        uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
        uint64_t curT)
    {
        // dav-m200 Cube does not commit partial M/N score tiles. Safe-gate
        // chunks also use this path because their long cumulative decay range
        // cannot be factorized through FP16 Cube operands without overflow.
        constexpr uint64_t fp32ArenaOffset = 1024;
        constexpr uint64_t partialStride = 8;
        constexpr uint64_t maxPartials = 4;
        LocalTensor<T> typedArena = vecBuf_.Get<T>();
        LocalTensor<T> rowVector = typedArena;
        LocalTensor<T> gRow = typedArena[K_];
        LocalTensor<T> kCol = typedArena[2 * K_];
        LocalTensor<T> gCol = typedArena[3 * K_];
        LocalTensor<T> gate = typedArena[4 * K_];
        LocalTensor<T> kGated = typedArena[5 * K_];
        LocalTensor<T> product = typedArena[6 * K_];

        LocalTensor<float> fp32Arena = vecBuf_.Get<float>()[fp32ArenaOffset];
        LocalTensor<float> productFp32 = fp32Arena;
        LocalTensor<float> partials = fp32Arena[K_];
        LocalTensor<float> scoreRow = partials[maxPartials * partialStride];

#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
        // Keep both original score passes and their FP16 operation order.
        // Only immutable key/gate columns are cached; no score arithmetic
        // or per-dot-product synchronization is fused by this experiment.
        // Reserve the subsequent triangular solve's UB scratch as well;
        // its highest live offset is below 72 KiB.
        constexpr uint64_t scoreCacheByteOffset = 80 * 1024;
        constexpr uint64_t scoreCacheMaxRows = 64;
        constexpr uint64_t scoreCacheMaxChannels = 256;
        const uint64_t cacheElements = curT * K_;
        const bool cacheColumns = BT_ == scoreCacheMaxRows &&
            curT == scoreCacheMaxRows &&
            K_ >= 16 && K_ <= scoreCacheMaxChannels && K_ % 16 == 0 &&
            scoreCacheByteOffset + 2 * cacheElements * sizeof(T) <=
                KDA_VEC_ARENA_ELEMENTS * sizeof(float);
        LocalTensor<T> cachedKeys = typedArena;
        LocalTensor<T> cachedGates = typedArena;
        if (cacheColumns) {
            cachedKeys = typedArena[scoreCacheByteOffset / sizeof(T)];
            cachedGates = cachedKeys[cacheElements];
            if (inputSequenceMajor_) {
                for (uint64_t col = 0; col < curT; ++col) {
                    LocalTensor<T> keyColumn = cachedKeys[col * K_];
                    CopyVectorIn(keyColumn, k_, QOffset(b, h, start + col, 0), K_);
                }
            } else {
                CopyVectorIn(cachedKeys, k_, QOffset(b, h, start, 0), cacheElements);
            }
            // Gates are head-major for both public query/key layouts.
            CopyVectorIn(cachedGates, gk_, KVOffset(b, hv, start, 0, K_), cacheElements);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
#endif

        // Run K*K and Q*K in separate passes. A single dav-m200 vector pass
        // can otherwise leave the second adjacent dot product unwritten,
        // even when its product and reduction buffers do not alias.
        for (uint64_t row = 0; row < curT; ++row) {
            Duplicate(scoreRow, 0.0f, static_cast<uint32_t>(BT_));
            PipeBarrier<PIPE_V>();

            CopyVectorIn(
                rowVector, k_, QOffset(b, h, start + row, 0), K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            CopyVectorIn(
                gRow, gk_, KVOffset(b, hv, start + row, 0, K_), K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

            for (uint64_t col = 0; col <= row; ++col) {
#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
                if (cacheColumns) {
                    kCol = cachedKeys[col * K_];
                    gCol = cachedGates[col * K_];
                } else {
#endif
                CopyVectorIn(
                    kCol, k_, QOffset(b, h, start + col, 0), K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                CopyVectorIn(
                    gCol, gk_, KVOffset(b, hv, start + col, 0, K_), K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
                }
#endif

                Sub(gate, gRow, gCol, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Muls(gate, gate, static_cast<T>(LN2),
                     static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                ClampFp16ExpInput(gate, static_cast<uint32_t>(K_));
                Exp(gate, gate, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Mul(kGated, kCol, gate, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Mul(product, rowVector, kGated, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Cast(productFp32, product, RoundMode::CAST_NONE,
                     static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                ReduceDotProduct310P(
                    scoreRow, col, productFp32, partials);
            }

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(akk_[AOffset(b, hv, start + row, 0)], scoreRow,
                     static_cast<uint32_t>(BT_));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }

        for (uint64_t row = 0; row < curT; ++row) {
            Duplicate(scoreRow, 0.0f, static_cast<uint32_t>(BT_));
            PipeBarrier<PIPE_V>();

            CopyVectorIn(
                rowVector, q_, QOffset(b, h, start + row, 0), K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            CopyVectorIn(
                gRow, gk_, KVOffset(b, hv, start + row, 0, K_), K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

            for (uint64_t col = 0; col <= row; ++col) {
#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
                if (cacheColumns) {
                    kCol = cachedKeys[col * K_];
                    gCol = cachedGates[col * K_];
                } else {
#endif
                CopyVectorIn(
                    kCol, k_, QOffset(b, h, start + col, 0), K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                CopyVectorIn(
                    gCol, gk_, KVOffset(b, hv, start + col, 0, K_), K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
                }
#endif

                Sub(gate, gRow, gCol, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Muls(gate, gate, static_cast<T>(LN2),
                     static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                ClampFp16ExpInput(gate, static_cast<uint32_t>(K_));
                Exp(gate, gate, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Mul(kGated, kCol, gate, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Mul(product, rowVector, kGated,
                    static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Cast(productFp32, product, RoundMode::CAST_NONE,
                     static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                ReduceDotProduct310P(
                    scoreRow, col, productFp32, partials);
            }

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(aqk_[AOffset(b, hv, start + row, 0)], scoreRow,
                     static_cast<uint32_t>(BT_));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }
};
extern "C" __global__ __aicore__ void glm_kda_score_probe_v1(
    GM_ADDR q,GM_ADDR k,GM_ADDR g,GM_ADDR aqk,GM_ADDR akk) {
    InitSocState();
    ScoreProbe<half> probe;
    probe.Run(q,k,g,aqk,akk);
}
