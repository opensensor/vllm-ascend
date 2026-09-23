#ifndef QSA_SPARSE_ATTENTION_V310_H
#define QSA_SPARSE_ATTENTION_V310_H

#include "kernel_operator.h"
#include "qsa_sparse_attention_v310_tiling_data.h"

namespace NsQsaSparseAttention {

using namespace AscendC;

constexpr int64_t NZ_INNER = 16;
constexpr int64_t QSA_COMPRESS_RATIO = 4;
constexpr int64_t SCALAR_VECTOR_WIDTH = 8;
constexpr int64_t FP32_REDUCE_WIDTH = 64;
constexpr float SCALE_Q24_FACTOR = 16777216.0f;

class QsaSparseAttentionV310 {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR groupIndices,
                                GM_ADDR groupCounts, GM_ADDR tailStarts, GM_ADDR tailCounts, GM_ADDR blockTable,
                                GM_ADDR queryStartLoc, GM_ADDR output, const QsaSparseAttentionV310TilingData *tiling,
                                TPipe *pipe)
    {
        numTokens_ = tiling->numTokens;
        numQueryHeads_ = tiling->numQueryHeads;
        numKvHeads_ = tiling->numKvHeads;
        headDim_ = tiling->headDim;
        cacheBlockSize_ = tiling->cacheBlockSize;
        cacheHeadDimBlocks_ = tiling->cacheHeadDimBlocks;
        maxBlocksPerSequence_ = tiling->maxBlocksPerSequence;
        selectedGroupsWidth_ = tiling->selectedGroupsWidth;
        numRequests_ = tiling->numRequests;
        tasksPerCore_ = tiling->tasksPerCore;
        taskCount_ = tiling->taskCount;
        // 310P custom-op tiling reliably transports integral fields. Carry
        // this scalar as Q24 so it cannot be lost by the float-field ABI.
        scale_ = static_cast<float>(tiling->scaleQ24) / SCALE_Q24_FACTOR;
        queryHeadsPerKvHead_ = numQueryHeads_ / numKvHeads_;
        headDimBlocks_ = headDim_ / NZ_INNER;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(valueCache));
        groupIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupIndices));
        groupCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupCounts));
        tailStartsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tailStarts));
        tailCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tailCounts));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        queryStartLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));

        pipe_ = pipe;
        pipe_->InitBuffer(halfQueryBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(halfKvBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(queryBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(kvBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(productBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(accumulatorBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, SCALAR_VECTOR_WIDTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t firstTask = GetBlockIdx() * tasksPerCore_;
        for (int64_t offset = 0; offset < tasksPerCore_; ++offset) {
            const int64_t task = firstTask + offset;
            if (task >= taskCount_) {
                break;
            }
            ComputeTask(task);
        }
    }

private:
    __aicore__ inline int64_t RequestForToken(int64_t token) const
    {
        // Number of requests is small (decode normally has one token/request).
        // Keeping this lookup on the AI core avoids a device-to-host sync or a
        // separate token-to-request materialization kernel.
        for (int64_t request = 0; request < numRequests_; ++request) {
            if (token < queryStartLocGm_.GetValue(request + 1)) {
                return request;
            }
        }
        return numRequests_ - 1;
    }

    __aicore__ inline int64_t CacheOffset(int64_t request, int64_t token, int64_t kvHead, int64_t dimBlock) const
    {
        const int64_t logicalBlock = token / cacheBlockSize_;
        const int64_t tokenOffset = token % cacheBlockSize_;
        const int64_t physicalBlock = blockTableGm_.GetValue(request * maxBlocksPerSequence_ + logicalBlock);
        const int64_t channelBlock = kvHead * headDimBlocks_ + dimBlock;
        return ((physicalBlock * cacheHeadDimBlocks_ + channelBlock) * cacheBlockSize_ + tokenOffset) * NZ_INNER;
    }

    __aicore__ inline void LoadNzRow(LocalTensor<half> dst, const GlobalTensor<half> &cache, int64_t request,
                                     int64_t token, int64_t kvHead)
    {
        for (int64_t dimBlock = 0; dimBlock < headDimBlocks_; ++dimBlock) {
            DataCopy(dst[dimBlock * NZ_INNER], cache[CacheOffset(request, token, kvHead, dimBlock)], NZ_INNER);
        }
    }

    __aicore__ inline float ExpScalar(float value, LocalTensor<float> scalar)
    {
        Duplicate(scalar, value, SCALAR_VECTOR_WIDTH);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, SCALAR_VECTOR_WIDTH);
        PipeBarrier<PIPE_V>();
        return scalar.GetValue(0);
    }

    __aicore__ inline float ReduceHead(LocalTensor<float> product, LocalTensor<float> scalar)
    {
        float sum = 0.0f;
        for (int64_t offset = 0; offset < headDim_; offset += FP32_REDUCE_WIDTH) {
            const int64_t width = headDim_ - offset < FP32_REDUCE_WIDTH
                                      ? headDim_ - offset : FP32_REDUCE_WIDTH;
            WholeReduceSum(scalar, product[offset], width, 1, 1, 1, SCALAR_VECTOR_WIDTH);
            PipeBarrier<PIPE_V>();
            sum += scalar.GetValue(0);
        }
        return sum;
    }

    __aicore__ inline void AccumulateToken(int64_t request, int64_t token, int64_t kvHead,
                                           LocalTensor<float> query, LocalTensor<half> halfKv,
                                           LocalTensor<float> kv, LocalTensor<float> product,
                                           LocalTensor<float> accumulator, LocalTensor<float> scalar,
                                           float &rowMax, float &rowSum)
    {
        LoadNzRow(halfKv, keyCacheGm_, request, token, kvHead);
        PipeBarrier<PIPE_ALL>();
        Cast(kv, halfKv, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        Mul(product, query, kv, headDim_);
        PipeBarrier<PIPE_V>();
        const float score = ReduceHead(product, scalar) * scale_;
        const float nextMax = score > rowMax ? score : rowMax;
        const float previousWeight = rowSum == 0.0f ? 0.0f : ExpScalar(rowMax - nextMax, scalar);
        const float tokenWeight = ExpScalar(score - nextMax, scalar);

        Muls(accumulator, accumulator, previousWeight, headDim_);
        PipeBarrier<PIPE_V>();
        LoadNzRow(halfKv, valueCacheGm_, request, token, kvHead);
        PipeBarrier<PIPE_ALL>();
        Cast(kv, halfKv, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        Muls(kv, kv, tokenWeight, headDim_);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, kv, headDim_);
        PipeBarrier<PIPE_V>();
        rowSum = rowSum * previousWeight + tokenWeight;
        rowMax = nextMax;
    }

    __aicore__ inline void ComputeTask(int64_t task)
    {
        const int64_t tokenRow = task / numQueryHeads_;
        const int64_t queryHead = task % numQueryHeads_;
        const int64_t kvHead = queryHead / queryHeadsPerKvHead_;
        const int64_t request = RequestForToken(tokenRow);

        LocalTensor<half> halfQuery = halfQueryBuf_.Get<half>();
        LocalTensor<half> halfKv = halfKvBuf_.Get<half>();
        LocalTensor<float> query = queryBuf_.Get<float>();
        LocalTensor<float> kv = kvBuf_.Get<float>();
        LocalTensor<float> product = productBuf_.Get<float>();
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();

        const int64_t queryOffset = (tokenRow * numQueryHeads_ + queryHead) * headDim_;
        DataCopy(halfQuery, queryGm_[queryOffset], headDim_);
        PipeBarrier<PIPE_ALL>();
        Cast(query, halfQuery, RoundMode::CAST_NONE, headDim_);
        Duplicate(accumulator, 0.0f, headDim_);
        PipeBarrier<PIPE_V>();

        float rowMax = -3.402823466e38f;
        float rowSum = 0.0f;
        const int64_t groupCount = groupCountsGm_.GetValue(tokenRow);
        for (int64_t groupRank = 0; groupRank < groupCount; ++groupRank) {
            const int64_t group = groupIndicesGm_.GetValue(tokenRow * selectedGroupsWidth_ + groupRank);
            const int64_t groupStart = group * QSA_COMPRESS_RATIO;
            for (int64_t inGroup = 0; inGroup < QSA_COMPRESS_RATIO; ++inGroup) {
                AccumulateToken(request, groupStart + inGroup, kvHead, query, halfKv, kv, product, accumulator,
                                scalar, rowMax, rowSum);
            }
        }
        const int64_t tailStart = tailStartsGm_.GetValue(tokenRow);
        const int64_t tailCount = tailCountsGm_.GetValue(tokenRow);
        for (int64_t tail = 0; tail < tailCount; ++tail) {
            AccumulateToken(request, tailStart + tail, kvHead, query, halfKv, kv, product, accumulator, scalar,
                            rowMax, rowSum);
        }

        if (rowSum > 0.0f) {
            Muls(accumulator, accumulator, 1.0f / rowSum, headDim_);
        }
        PipeBarrier<PIPE_V>();
        Cast(halfKv, accumulator, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outputGm_[queryOffset], halfKv, headDim_);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> halfQueryBuf_;
    TBuf<TPosition::VECCALC> halfKvBuf_;
    TBuf<TPosition::VECCALC> queryBuf_;
    TBuf<TPosition::VECCALC> kvBuf_;
    TBuf<TPosition::VECCALC> productBuf_;
    TBuf<TPosition::VECCALC> accumulatorBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    GlobalTensor<half> queryGm_;
    GlobalTensor<half> keyCacheGm_;
    GlobalTensor<half> valueCacheGm_;
    GlobalTensor<int32_t> groupIndicesGm_;
    GlobalTensor<int32_t> groupCountsGm_;
    GlobalTensor<int32_t> tailStartsGm_;
    GlobalTensor<int32_t> tailCountsGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<half> outputGm_;
    int64_t numTokens_ = 0;
    int64_t numQueryHeads_ = 0;
    int64_t numKvHeads_ = 0;
    int64_t headDim_ = 0;
    int64_t cacheBlockSize_ = 0;
    int64_t cacheHeadDimBlocks_ = 0;
    int64_t maxBlocksPerSequence_ = 0;
    int64_t selectedGroupsWidth_ = 0;
    int64_t numRequests_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t taskCount_ = 0;
    int64_t queryHeadsPerKvHead_ = 0;
    int64_t headDimBlocks_ = 0;
    float scale_ = 1.0f;
};

}  // namespace NsQsaSparseAttention

#endif
