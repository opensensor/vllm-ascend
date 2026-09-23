#ifndef QSA_INDEXER_SCORE_V310_H
#define QSA_INDEXER_SCORE_V310_H

#include "kernel_operator.h"
#include "qsa_indexer_score_v310_tiling_data.h"

namespace NsQsaIndexerScore {

using namespace AscendC;

constexpr int64_t QSA_COMPRESS_RATIO = 4;
constexpr int64_t FP32_REDUCE_WIDTH = 64;

class QsaIndexerScoreV310 {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR compressedKeyCache, GM_ADDR blockTable,
                                GM_ADDR queryStartLoc, GM_ADDR positions, GM_ADDR scores,
                                const QsaIndexerScoreV310TilingData *tiling, TPipe *pipe)
    {
        numHeads_ = tiling->numHeads;
        headDim_ = tiling->headDim;
        cacheRowsPerBlock_ = tiling->cacheRowsPerBlock;
        groupsPerBlock_ = tiling->groupsPerBlock;
        maxBlocksPerSequence_ = tiling->maxBlocksPerSequence;
        maxGroupsPerSequence_ = tiling->maxGroupsPerSequence;
        numRequests_ = tiling->numRequests;
        tasksPerCore_ = tiling->tasksPerCore;
        taskCount_ = tiling->taskCount;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query));
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(compressedKeyCache));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        queryStartLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
        positionsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(positions));
        scoresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));
        pipe_ = pipe;
        pipe_->InitBuffer(queryHalfBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(keyHalfBuf_, headDim_ * sizeof(half));
        pipe_->InitBuffer(queryFloatBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(keyFloatBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(productBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t firstTask = GetBlockIdx() * tasksPerCore_;
        for (int64_t offset = 0; offset < tasksPerCore_; ++offset) {
            const int64_t task = firstTask + offset;
            if (task >= taskCount_) {
                break;
            }
            Compute(task);
        }
    }

private:
    __aicore__ inline float ReduceHead(LocalTensor<float> product, LocalTensor<float> reduce)
    {
        float sum = 0.0f;
        for (int64_t offset = 0; offset < headDim_; offset += FP32_REDUCE_WIDTH) {
            const int64_t width = headDim_ - offset < FP32_REDUCE_WIDTH
                                      ? headDim_ - offset : FP32_REDUCE_WIDTH;
            WholeReduceSum(reduce, product[offset], width, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            sum += reduce.GetValue(0);
        }
        return sum;
    }

    __aicore__ inline int64_t RequestForToken(int64_t row) const
    {
        for (int64_t request = 0; request < numRequests_; ++request) {
            if (row < queryStartLocGm_.GetValue(request + 1)) {
                return request;
            }
        }
        return numRequests_ - 1;
    }

    __aicore__ inline void Compute(int64_t task)
    {
        const int64_t row = task / maxGroupsPerSequence_;
        const int64_t group = task % maxGroupsPerSequence_;
        const int64_t visibleGroups = (static_cast<int64_t>(positionsGm_.GetValue(row)) + 1) / QSA_COMPRESS_RATIO;
        if (group >= visibleGroups) {
            scoresGm_.SetValue(task, -3.402823466e38f);
            return;
        }
        const int64_t request = RequestForToken(row);
        const int64_t logicalBlock = group / groupsPerBlock_;
        const int64_t rowInBlock = group % groupsPerBlock_;
        const int64_t physicalBlock =
            blockTableGm_.GetValue(request * maxBlocksPerSequence_ + logicalBlock);
        const int64_t keyOffset = (physicalBlock * cacheRowsPerBlock_ + rowInBlock) * headDim_;

        LocalTensor<half> queryHalf = queryHalfBuf_.Get<half>();
        LocalTensor<half> keyHalf = keyHalfBuf_.Get<half>();
        LocalTensor<float> queryFloat = queryFloatBuf_.Get<float>();
        LocalTensor<float> keyFloat = keyFloatBuf_.Get<float>();
        LocalTensor<float> product = productBuf_.Get<float>();
        LocalTensor<float> reduce = reduceBuf_.Get<float>();
        DataCopy(keyHalf, cacheGm_[keyOffset], headDim_);
        PipeBarrier<PIPE_ALL>();
        Cast(keyFloat, keyHalf, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();

        float score = 0.0f;
        for (int64_t head = 0; head < numHeads_; ++head) {
            const int64_t queryOffset = (row * numHeads_ + head) * headDim_;
            DataCopy(queryHalf, queryGm_[queryOffset], headDim_);
            PipeBarrier<PIPE_ALL>();
            Cast(queryFloat, queryHalf, RoundMode::CAST_NONE, headDim_);
            PipeBarrier<PIPE_V>();
            Mul(product, queryFloat, keyFloat, headDim_);
            PipeBarrier<PIPE_V>();
            const float dot = ReduceHead(product, reduce);
            score += dot > 0.0f ? dot : 0.0f;
        }
        scoresGm_.SetValue(task, score);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> queryHalfBuf_;
    TBuf<TPosition::VECCALC> keyHalfBuf_;
    TBuf<TPosition::VECCALC> queryFloatBuf_;
    TBuf<TPosition::VECCALC> keyFloatBuf_;
    TBuf<TPosition::VECCALC> productBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    GlobalTensor<half> queryGm_;
    GlobalTensor<half> cacheGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<int32_t> positionsGm_;
    GlobalTensor<float> scoresGm_;
    int64_t numHeads_ = 0;
    int64_t headDim_ = 0;
    int64_t cacheRowsPerBlock_ = 0;
    int64_t groupsPerBlock_ = 0;
    int64_t maxBlocksPerSequence_ = 0;
    int64_t maxGroupsPerSequence_ = 0;
    int64_t numRequests_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t taskCount_ = 0;
};

}  // namespace NsQsaIndexerScore

#endif
