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
constexpr int64_t MAX_HEAD_DIM = 256;
constexpr int64_t MAX_QUERY_HEADS_PER_KV_HEAD = 24;
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
        headsPerTask_ = tiling->headsPerTask;
        taskTilesPerKvHead_ = tiling->taskTilesPerKvHead;
        totalQueryHeadsPerKvHead_ = numQueryHeads_ / numKvHeads_;
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
        queryHeadsPerKvHead_ = headsPerTask_;
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
        pipe_->InitBuffer(halfQueryBuf_, queryHeadsPerKvHead_ * headDim_ * sizeof(half));
        pipe_->InitBuffer(halfKvBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(queryBuf_, queryHeadsPerKvHead_ * headDim_ * sizeof(float));
        pipe_->InitBuffer(kvBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(productBuf_, queryHeadsPerKvHead_ * headDim_ * sizeof(float));
        pipe_->InitBuffer(accumulatorBuf_, queryHeadsPerKvHead_ * headDim_ * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 2 * SCALAR_VECTOR_WIDTH * sizeof(float));
        pipe_->InitBuffer(exponentBuf_, MAX_QUERY_HEADS_PER_KV_HEAD * sizeof(float));
        pipe_->InitBuffer(partialScoreBuf_, MAX_QUERY_HEADS_PER_KV_HEAD * 4 * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, 2 * MAX_QUERY_HEADS_PER_KV_HEAD * sizeof(float));
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

    __aicore__ inline void LoadNzRow(LocalTensor<half> dst, const GlobalTensor<half> &cache,
                                     int64_t firstBlockOffset)
    {
        // Each NZ channel block contributes one contiguous 32-byte row. A
        // strided MTE2 transfer replaces one tiny GM copy per channel block.
        const DataCopyParams copyParams{static_cast<uint16_t>(headDimBlocks_), 1,
                                        static_cast<uint16_t>(cacheBlockSize_ - 1), 0};
        DataCopy(dst, cache[firstBlockOffset], copyParams);
    }

    __aicore__ inline float ReduceHead(LocalTensor<float> product, LocalTensor<float> scalar)
    {
        if (headDim_ % FP32_REDUCE_WIDTH == 0 && headDim_ > FP32_REDUCE_WIDTH) {
            const int64_t repeats = headDim_ / FP32_REDUCE_WIDTH;
            WholeReduceSum(scalar, product, FP32_REDUCE_WIDTH, repeats, 1, 1, SCALAR_VECTOR_WIDTH);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(scalar[SCALAR_VECTOR_WIDTH], scalar, repeats, 1, 1, 1, SCALAR_VECTOR_WIDTH);
            PipeBarrier<PIPE_V>();
            return scalar.GetValue(SCALAR_VECTOR_WIDTH);
        }
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

    __aicore__ inline void AccumulateToken(int64_t firstBlockOffset, LocalTensor<float> query,
                                           LocalTensor<half> halfKv,
                                           LocalTensor<float> kv, LocalTensor<float> product,
                                           LocalTensor<float> accumulator, LocalTensor<float> scalar,
                                           LocalTensor<float> exponents, LocalTensor<float> partialScores,
                                           LocalTensor<float> reducedScores,
                                           float *rowMax, float *rowSum, float *tokenWeight)
    {
        LoadNzRow(halfKv, keyCacheGm_, firstBlockOffset);
        PipeBarrier<PIPE_ALL>();
        Cast(kv, halfKv, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        if (headDim_ == MAX_HEAD_DIM) {
            // Reduce all 256-wide Q·K rows together. The first reduction
            // produces four adjacent partials per head; two strided reductions
            // combine even and odd heads without a per-head vector launch.
            for (int64_t head = 0; head < queryHeadsPerKvHead_; ++head) {
                const int64_t headOffset = head * headDim_;
                Mul(product[headOffset], query[headOffset], kv, headDim_);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(partialScores, product, FP32_REDUCE_WIDTH, queryHeadsPerKvHead_ * 4, 1, 1,
                           SCALAR_VECTOR_WIDTH);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(reducedScores, partialScores, 4, (queryHeadsPerKvHead_ + 1) / 2, 1, 1, 1);
            if (queryHeadsPerKvHead_ > 1) {
                WholeReduceSum(reducedScores[MAX_QUERY_HEADS_PER_KV_HEAD], partialScores[4], 4,
                               queryHeadsPerKvHead_ / 2, 1, 1, 1);
            }
            PipeBarrier<PIPE_V>();
        }
        float scores[MAX_QUERY_HEADS_PER_KV_HEAD];
        bool raisesMax[MAX_QUERY_HEADS_PER_KV_HEAD];
        for (int64_t head = 0; head < queryHeadsPerKvHead_; ++head) {
            const int64_t headOffset = head * headDim_;
            float score;
            if (headDim_ == MAX_HEAD_DIM) {
                const int64_t scoreOffset = head % 2 == 0 ? 0 : MAX_QUERY_HEADS_PER_KV_HEAD;
                score = reducedScores.GetValue(scoreOffset + head / 2) * scale_;
            } else {
                Mul(product, query[headOffset], kv, headDim_);
                PipeBarrier<PIPE_V>();
                score = ReduceHead(product, scalar) * scale_;
            }
            scores[head] = score;
            raisesMax[head] = score > rowMax[head];
            const float delta = rowSum[head] == 0.0f ? 0.0f
                : (raisesMax[head] ? rowMax[head] - score : score - rowMax[head]);
            exponents.SetValue(head, delta);
        }
        const int64_t expWidth = ((queryHeadsPerKvHead_ + SCALAR_VECTOR_WIDTH - 1) / SCALAR_VECTOR_WIDTH)
                                 * SCALAR_VECTOR_WIDTH;
        PipeBarrier<PIPE_V>();
        Exp(exponents, exponents, expWidth);
        PipeBarrier<PIPE_V>();
        for (int64_t head = 0; head < queryHeadsPerKvHead_; ++head) {
            const int64_t headOffset = head * headDim_;
            const float expWeight = exponents.GetValue(head);
            float previousWeight = 1.0f;
            tokenWeight[head] = 1.0f;
            if (rowSum[head] == 0.0f) {
                previousWeight = 0.0f;
            } else if (raisesMax[head]) {
                previousWeight = expWeight;
                Muls(accumulator[headOffset], accumulator[headOffset], previousWeight, headDim_);
                PipeBarrier<PIPE_V>();
            } else {
                tokenWeight[head] = expWeight;
            }
            rowSum[head] = rowSum[head] * previousWeight + tokenWeight[head];
            rowMax[head] = raisesMax[head] ? scores[head] : rowMax[head];
        }
        LoadNzRow(halfKv, valueCacheGm_, firstBlockOffset);
        PipeBarrier<PIPE_ALL>();
        Cast(kv, halfKv, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        for (int64_t head = 0; head < queryHeadsPerKvHead_; ++head) {
            const int64_t headOffset = head * headDim_;
            Axpy(accumulator[headOffset], kv, tokenWeight[head], headDim_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeTask(int64_t task)
    {
        const int64_t tokenRow = task / (numKvHeads_ * taskTilesPerKvHead_);
        const int64_t kvHeadTask = task % (numKvHeads_ * taskTilesPerKvHead_);
        const int64_t kvHead = kvHeadTask / taskTilesPerKvHead_;
        const int64_t headStart = (kvHeadTask % taskTilesPerKvHead_) * headsPerTask_;
        queryHeadsPerKvHead_ = totalQueryHeadsPerKvHead_ - headStart < headsPerTask_
                                   ? totalQueryHeadsPerKvHead_ - headStart : headsPerTask_;
        const int64_t request = RequestForToken(tokenRow);

        LocalTensor<half> halfQuery = halfQueryBuf_.Get<half>();
        LocalTensor<half> halfKv = halfKvBuf_.Get<half>();
        LocalTensor<float> query = queryBuf_.Get<float>();
        LocalTensor<float> kv = kvBuf_.Get<float>();
        LocalTensor<float> product = productBuf_.Get<float>();
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> exponents = exponentBuf_.Get<float>();
        LocalTensor<float> partialScores = partialScoreBuf_.Get<float>();
        LocalTensor<float> reducedScores = scoreBuf_.Get<float>();

        const int64_t queryOffset =
            (tokenRow * numQueryHeads_ + kvHead * totalQueryHeadsPerKvHead_ + headStart) * headDim_;
        const int64_t queryElements = queryHeadsPerKvHead_ * headDim_;
        DataCopy(halfQuery, queryGm_[queryOffset], queryElements);
        PipeBarrier<PIPE_ALL>();
        Cast(query, halfQuery, RoundMode::CAST_NONE, queryElements);
        Duplicate(accumulator, 0.0f, queryElements);
        Duplicate(exponents, 0.0f, MAX_QUERY_HEADS_PER_KV_HEAD);
        PipeBarrier<PIPE_V>();

        float rowMax[MAX_QUERY_HEADS_PER_KV_HEAD];
        float rowSum[MAX_QUERY_HEADS_PER_KV_HEAD];
        float tokenWeight[MAX_QUERY_HEADS_PER_KV_HEAD];
        for (int64_t head = 0; head < queryHeadsPerKvHead_; ++head) {
            rowMax[head] = -3.402823466e38f;
            rowSum[head] = 0.0f;
        }
        const int64_t groupCount = groupCountsGm_.GetValue(tokenRow);
        const int64_t channelBlock = kvHead * headDimBlocks_;
        for (int64_t groupRank = 0; groupRank < groupCount; ++groupRank) {
            const int64_t group = groupIndicesGm_.GetValue(tokenRow * selectedGroupsWidth_ + groupRank);
            const int64_t groupStart = group * QSA_COMPRESS_RATIO;
            // A compression group stays within one cache page because the
            // page size is a multiple of four. Resolve its page only once.
            const int64_t logicalBlock = groupStart / cacheBlockSize_;
            const int64_t physicalBlock = blockTableGm_.GetValue(request * maxBlocksPerSequence_ + logicalBlock);
            const int64_t firstBlockOffset =
                ((physicalBlock * cacheHeadDimBlocks_ + channelBlock) * cacheBlockSize_ +
                 groupStart % cacheBlockSize_) * NZ_INNER;
            for (int64_t inGroup = 0; inGroup < QSA_COMPRESS_RATIO; ++inGroup) {
                AccumulateToken(firstBlockOffset + inGroup * NZ_INNER, query, halfKv, kv, product, accumulator,
                                scalar, exponents, partialScores, reducedScores,
                                rowMax, rowSum, tokenWeight);
            }
        }
        const int64_t tailStart = tailStartsGm_.GetValue(tokenRow);
        const int64_t tailCount = tailCountsGm_.GetValue(tokenRow);
        for (int64_t tail = 0; tail < tailCount; ++tail) {
            const int64_t token = tailStart + tail;
            const int64_t logicalBlock = token / cacheBlockSize_;
            const int64_t physicalBlock = blockTableGm_.GetValue(request * maxBlocksPerSequence_ + logicalBlock);
            const int64_t firstBlockOffset =
                ((physicalBlock * cacheHeadDimBlocks_ + channelBlock) * cacheBlockSize_ +
                 token % cacheBlockSize_) * NZ_INNER;
            AccumulateToken(firstBlockOffset, query, halfKv, kv, product, accumulator, scalar,
                            exponents, partialScores, reducedScores, rowMax, rowSum, tokenWeight);
        }

        for (int64_t head = 0; head < queryHeadsPerKvHead_; ++head) {
            if (rowSum[head] > 0.0f) {
                Muls(accumulator[head * headDim_], accumulator[head * headDim_], 1.0f / rowSum[head], headDim_);
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(halfQuery, accumulator, RoundMode::CAST_NONE, queryElements);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outputGm_[queryOffset], halfQuery, queryElements);
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
    TBuf<TPosition::VECCALC> exponentBuf_;
    TBuf<TPosition::VECCALC> partialScoreBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
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
    int64_t headsPerTask_ = 0;
    int64_t taskTilesPerKvHead_ = 0;
    int64_t totalQueryHeadsPerKvHead_ = 0;
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
