#ifndef QSA_INDEX_CACHE_UPDATE_V310_H
#define QSA_INDEX_CACHE_UPDATE_V310_H

#include "kernel_operator.h"
#include "qsa_index_cache_update_v310_tiling_data.h"

namespace NsQsaIndexCacheUpdate {

using namespace AscendC;

constexpr int64_t QSA_COMPRESS_RATIO = 4;
constexpr int64_t FP32_REDUCE_WIDTH = 64;

class QsaIndexCacheUpdateV310 {
public:
    __aicore__ inline void Init(GM_ADDR compressedKeyCache, GM_ADDR indexKeys,
                                GM_ADDR queryStartLoc, GM_ADDR slotMapping,
                                GM_ADDR keyNormWeight, GM_ADDR ropeCos, GM_ADDR ropeSin,
                                GM_ADDR compressedKeyCacheOut,
                                const QsaIndexCacheUpdateV310TilingData *tiling, TPipe *pipe)
    {
        headDim_ = tiling->headDim;
        cacheRowsPerBlock_ = tiling->cacheRowsPerBlock;
        groupsPerBlock_ = tiling->groupsPerBlock;
        blockSize_ = tiling->blockSize;
        rotaryDim_ = tiling->rotaryDim;
        numRequests_ = tiling->numRequests;
        requestsPerCore_ = tiling->requestsPerCore;
        normEps_ = tiling->normEps;
        cacheInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(compressedKeyCache));
        keysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(indexKeys));
        queryStartLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
        slotMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotMapping));
        keyNormWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(keyNormWeight));
        ropeCosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ropeCos));
        ropeSinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ropeSin));
        cacheOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(compressedKeyCacheOut));
        pipe_ = pipe;
        pipe_->InitBuffer(rawHalfBuf_, QSA_COMPRESS_RATIO * headDim_ * sizeof(half));
        pipe_->InitBuffer(rawFloatBuf_, QSA_COMPRESS_RATIO * headDim_ * sizeof(float));
        pipe_->InitBuffer(sumFloatBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(meanHalfBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(workFloatBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(normWeightHalfBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(normWeightFloatBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t firstRequest = GetBlockIdx() * requestsPerCore_;
        for (int64_t offset = 0; offset < requestsPerCore_; ++offset) {
            const int64_t request = firstRequest + offset;
            if (request >= numRequests_) {
                break;
            }
            const int64_t start = queryStartLocGm_.GetValue(request);
            const int64_t end = queryStartLocGm_.GetValue(request + 1);
            for (int64_t row = start; row < end; ++row) {
                Update(row);
            }
        }
    }

private:
    __aicore__ inline float ReduceHead(LocalTensor<float> values, LocalTensor<float> reduce)
    {
        float sum = 0.0f;
        for (int64_t offset = 0; offset < headDim_; offset += FP32_REDUCE_WIDTH) {
            const int64_t width = headDim_ - offset < FP32_REDUCE_WIDTH
                                      ? headDim_ - offset : FP32_REDUCE_WIDTH;
            WholeReduceSum(reduce, values[offset], width, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            sum += reduce.GetValue(0);
        }
        return sum;
    }

    __aicore__ inline void Update(int64_t row)
    {
        const int64_t slot = slotMappingGm_.GetValue(row);
        if (slot < 0) {
            return;
        }
        const int64_t physicalBlock = slot / blockSize_;
        const int64_t tokenOffset = slot % blockSize_;
        const int64_t groupRow = tokenOffset / QSA_COMPRESS_RATIO;
        const int64_t groupOffset = tokenOffset % QSA_COMPRESS_RATIO;
        const int64_t keyOffset = row * headDim_;

        if (groupOffset < QSA_COMPRESS_RATIO - 1) {
            const int64_t scratchRow = groupsPerBlock_ + groupOffset;
            const int64_t cacheOffset =
                (physicalBlock * cacheRowsPerBlock_ + scratchRow) * headDim_;
            LocalTensor<half> meanHalf = meanHalfBuf_.Get<half>();
            DataCopy(meanHalf, keysGm_[keyOffset], headDim_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(cacheOutGm_[cacheOffset], meanHalf, headDim_);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        LocalTensor<half> rawHalf = rawHalfBuf_.Get<half>();
        LocalTensor<float> rawFloat = rawFloatBuf_.Get<float>();
        LocalTensor<float> sumFloat = sumFloatBuf_.Get<float>();
        LocalTensor<half> meanHalf = meanHalfBuf_.Get<half>();
        LocalTensor<float> workFloat = workFloatBuf_.Get<float>();
        LocalTensor<half> normWeightHalf = normWeightHalfBuf_.Get<half>();
        LocalTensor<float> normWeightFloat = normWeightFloatBuf_.Get<float>();
        LocalTensor<float> reduce = reduceBuf_.Get<float>();
        for (int64_t scratch = 0; scratch < QSA_COMPRESS_RATIO - 1; ++scratch) {
            const int64_t scratchOffset =
                (physicalBlock * cacheRowsPerBlock_ + groupsPerBlock_ + scratch) * headDim_;
            DataCopy(rawHalf[scratch * headDim_], cacheOutGm_[scratchOffset], headDim_);
        }
        DataCopy(rawHalf[(QSA_COMPRESS_RATIO - 1) * headDim_], keysGm_[keyOffset], headDim_);
        PipeBarrier<PIPE_ALL>();
        Cast(rawFloat, rawHalf, RoundMode::CAST_NONE, QSA_COMPRESS_RATIO * headDim_);
        PipeBarrier<PIPE_V>();
        Adds(sumFloat, rawFloat, 0.0f, headDim_);
        for (int64_t source = 1; source < QSA_COMPRESS_RATIO; ++source) {
            Add(sumFloat, sumFloat, rawFloat[source * headDim_], headDim_);
            PipeBarrier<PIPE_V>();
        }
        Muls(sumFloat, sumFloat, 1.0f / static_cast<float>(QSA_COMPRESS_RATIO), headDim_);
        PipeBarrier<PIPE_V>();

        // The model contract pools raw index keys first, then applies Gemma
        // RMSNorm and the first token's Neox-style RoPE to the complete group.
        Mul(workFloat, sumFloat, sumFloat, headDim_);
        PipeBarrier<PIPE_V>();
        const float variance = ReduceHead(workFloat, reduce) / static_cast<float>(headDim_) + normEps_;
        Duplicate(reduce, variance, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(reduce, reduce, 8);
        PipeBarrier<PIPE_V>();
        Muls(sumFloat, sumFloat, 1.0f / reduce.GetValue(0), headDim_);
        DataCopy(normWeightHalf, keyNormWeightGm_, headDim_);
        PipeBarrier<PIPE_ALL>();
        Cast(normWeightFloat, normWeightHalf, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        Adds(normWeightFloat, normWeightFloat, 1.0f, headDim_);
        PipeBarrier<PIPE_V>();
        Mul(sumFloat, sumFloat, normWeightFloat, headDim_);
        PipeBarrier<PIPE_V>();

        const int64_t halfRotaryDim = rotaryDim_ / 2;
        const int64_t ropeOffset = row * rotaryDim_;
        for (int64_t dim = 0; dim < halfRotaryDim; ++dim) {
            const float first = sumFloat.GetValue(dim);
            const float second = sumFloat.GetValue(dim + halfRotaryDim);
            const float cosFirst = static_cast<float>(ropeCosGm_.GetValue(ropeOffset + dim));
            const float sinFirst = static_cast<float>(ropeSinGm_.GetValue(ropeOffset + dim));
            const float cosSecond = static_cast<float>(ropeCosGm_.GetValue(ropeOffset + dim + halfRotaryDim));
            const float sinSecond = static_cast<float>(ropeSinGm_.GetValue(ropeOffset + dim + halfRotaryDim));
            sumFloat.SetValue(dim, first * cosFirst - second * sinFirst);
            sumFloat.SetValue(dim + halfRotaryDim, second * cosSecond + first * sinSecond);
        }
        Cast(meanHalf, sumFloat, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_ALL>();
        const int64_t outputOffset =
            (physicalBlock * cacheRowsPerBlock_ + groupRow) * headDim_;
        DataCopy(cacheOutGm_[outputOffset], meanHalf, headDim_);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rawHalfBuf_;
    TBuf<TPosition::VECCALC> rawFloatBuf_;
    TBuf<TPosition::VECCALC> sumFloatBuf_;
    TBuf<TPosition::VECCALC> meanHalfBuf_;
    TBuf<TPosition::VECCALC> workFloatBuf_;
    TBuf<TPosition::VECCALC> normWeightHalfBuf_;
    TBuf<TPosition::VECCALC> normWeightFloatBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    GlobalTensor<half> cacheInGm_;
    GlobalTensor<half> keysGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<half> keyNormWeightGm_;
    GlobalTensor<half> ropeCosGm_;
    GlobalTensor<half> ropeSinGm_;
    GlobalTensor<half> cacheOutGm_;
    int64_t headDim_ = 0;
    int64_t cacheRowsPerBlock_ = 0;
    int64_t groupsPerBlock_ = 0;
    int64_t blockSize_ = 0;
    int64_t rotaryDim_ = 0;
    int64_t numRequests_ = 0;
    int64_t requestsPerCore_ = 0;
    float normEps_ = 0.0f;
};

}  // namespace NsQsaIndexCacheUpdate

#endif
