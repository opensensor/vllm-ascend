#ifndef QSA_CUBE_SPARSE_ATTENTION_V310_H
#define QSA_CUBE_SPARSE_ATTENTION_V310_H

#include "kernel_operator.h"
#include "qsa_sparse_attention_v310_tiling_data.h"

namespace NsQsaCubeSparseAttention {

using namespace AscendC;

constexpr int64_t NZ_INNER = 16;
constexpr int64_t QSA_COMPRESS_RATIO = 4;
// A 64-token tile leaves enough UB for the transposed key operand while K and
// V continue to reuse the same B1/B2 buffers.
constexpr int64_t TOKEN_TILE = 64;
constexpr int64_t MAX_HEAD_DIM = 256;
constexpr int64_t MAX_QUERY_HEADS = 16;
constexpr int64_t MAX_GROUP_WIDTH = 512;
constexpr int64_t MAX_LOCAL_BLOCK_TABLE_ENTRIES = 2048;
constexpr int64_t INT32_ALIGNMENT = 8;
constexpr int64_t SCALE_Q24_FACTOR = 16777216;
constexpr int64_t QUERY_L0_ELEMENTS = MAX_QUERY_HEADS * MAX_HEAD_DIM;
constexpr int64_t PROBABILITY_L0_ELEMENTS = MAX_QUERY_HEADS * TOKEN_TILE;
constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};

class QsaCubeSparseAttentionV310 {
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
        headDimBlocks_ = headDim_ / NZ_INNER;
        cacheBlockSize_ = tiling->cacheBlockSize;
        cacheHeadDimBlocks_ = tiling->cacheHeadDimBlocks;
        maxBlocksPerSequence_ = tiling->maxBlocksPerSequence;
        selectedGroupsWidth_ = tiling->selectedGroupsWidth;
        numRequests_ = tiling->numRequests;
        tasksPerCore_ = tiling->tasksPerCore;
        taskCount_ = tiling->taskCount;
        scale_ = static_cast<float>(tiling->scaleQ24) / static_cast<float>(SCALE_Q24_FACTOR);
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

        useLocalBlockTable_ = numRequests_ == 1 && maxBlocksPerSequence_ > 0 &&
                              maxBlocksPerSequence_ <= MAX_LOCAL_BLOCK_TABLE_ENTRIES &&
                              maxBlocksPerSequence_ % INT32_ALIGNMENT == 0;
        useLocalGroups_ = selectedGroupsWidth_ > 0 && selectedGroupsWidth_ <= MAX_GROUP_WIDTH &&
                          selectedGroupsWidth_ % INT32_ALIGNMENT == 0;

        pipe_ = pipe;
        pipe_->InitBuffer(queryL1Buf_, MAX_QUERY_HEADS * MAX_HEAD_DIM * sizeof(half));
        pipe_->InitBuffer(probabilityL1Buf_, MAX_QUERY_HEADS * TOKEN_TILE * sizeof(half));
        pipe_->InitBuffer(kvL1Buf_, TOKEN_TILE * MAX_HEAD_DIM * sizeof(half));
        pipe_->InitBuffer(aL0Buf_, (QUERY_L0_ELEMENTS + PROBABILITY_L0_ELEMENTS) * sizeof(half));
        pipe_->InitBuffer(bL0Buf_, TOKEN_TILE * MAX_HEAD_DIM * sizeof(half));
        pipe_->InitBuffer(cL0Buf_, MAX_QUERY_HEADS * MAX_HEAD_DIM * sizeof(float));
        // Gather through UB before copying into B1.  Scattered GM -> B1 DMA is
        // much slower on 310P than the vector MTE2 path used by the standalone
        // NZ gather operator.  This keeps the staging entirely on chip: the
        // selected K/V tile is never written to and reread from global memory.
        pipe_->InitBuffer(kvGatherBuf_, TOKEN_TILE * MAX_HEAD_DIM * sizeof(half));
        pipe_->InitBuffer(kvTransposeBuf_, TOKEN_TILE * MAX_HEAD_DIM * sizeof(half));
        pipe_->InitBuffer(scoreBuf_, MAX_QUERY_HEADS * TOKEN_TILE * sizeof(float));
        pipe_->InitBuffer(softmaxBuf_, MAX_QUERY_HEADS * TOKEN_TILE * sizeof(float));
        pipe_->InitBuffer(probabilityBuf_, MAX_QUERY_HEADS * TOKEN_TILE * sizeof(half));
        pipe_->InitBuffer(contributionBuf_, MAX_QUERY_HEADS * MAX_HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(accumulatorBuf_, MAX_QUERY_HEADS * MAX_HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(outputBuf_, MAX_QUERY_HEADS * MAX_HEAD_DIM * sizeof(half));
        pipe_->InitBuffer(reduceBuf_, 2 * INT32_ALIGNMENT * sizeof(float));
        if (useLocalGroups_) {
            pipe_->InitBuffer(groupBuf_, selectedGroupsWidth_ * sizeof(int32_t));
        }
        if (useLocalBlockTable_) {
            pipe_->InitBuffer(blockTableBuf_, maxBlocksPerSequence_ * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        SetMMLayoutTransform(true);
        LocalTensor<int32_t> localBlockTable;
        if (useLocalBlockTable_) {
            localBlockTable = blockTableBuf_.Get<int32_t>();
            DataCopy(localBlockTable, blockTableGm_, maxBlocksPerSequence_);
            PipeBarrier<PIPE_ALL>();
        }
        const int64_t firstTask = GetBlockIdx() * tasksPerCore_;
        for (int64_t offset = 0; offset < tasksPerCore_; ++offset) {
            const int64_t task = firstTask + offset;
            if (task >= taskCount_) {
                break;
            }
            ComputeTask(task, localBlockTable);
        }
        SetMMLayoutTransform(false);
    }

private:
    __aicore__ inline int64_t RequestForToken(int64_t token) const
    {
        for (int64_t request = 0; request < numRequests_; ++request) {
            if (token < queryStartLocGm_.GetValue(request + 1)) {
                return request;
            }
        }
        return numRequests_ - 1;
    }

    __aicore__ inline int64_t PhysicalBlock(int64_t request, int64_t logicalBlock,
                                            const LocalTensor<int32_t> &localBlockTable) const
    {
        if (useLocalBlockTable_) {
            return localBlockTable.GetValue(logicalBlock);
        }
        return blockTableGm_.GetValue(request * maxBlocksPerSequence_ + logicalBlock);
    }

    __aicore__ inline int64_t GroupAt(int64_t tokenRow, int64_t groupRank, bool densePrefix,
                                      const LocalTensor<int32_t> &localGroups) const
    {
        if (densePrefix) {
            return groupRank;
        }
        return useLocalGroups_ ? localGroups.GetValue(groupRank)
                               : groupIndicesGm_.GetValue(tokenRow * selectedGroupsWidth_ + groupRank);
    }

    __aicore__ inline int64_t CacheOffset(int64_t request, int64_t token, int64_t kvHead,
                                          const LocalTensor<int32_t> &localBlockTable) const
    {
        const int64_t logicalBlock = token / cacheBlockSize_;
        const int64_t physicalBlock = PhysicalBlock(request, logicalBlock, localBlockTable);
        return ((physicalBlock * cacheHeadDimBlocks_ + kvHead * headDimBlocks_) * cacheBlockSize_ +
                token % cacheBlockSize_) * NZ_INNER;
    }

    __aicore__ inline void LoadQuery(int64_t queryOffset, int64_t queryHeads)
    {
        LocalTensor<half> queryL1 = queryL1Buf_.Get<half>();
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = queryHeads;
        params.dValue = headDim_;
        params.srcDValue = headDim_;
        params.dstNzC0Stride = MAX_QUERY_HEADS;
        params.dstNzNStride = 1;
        params.srcNdMatrixStride = 0;
        params.dstNzMatrixStride = 0;
        DataCopy(queryL1, queryGm_[queryOffset], params);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void LoadQueryToL0(int64_t queryHeads)
    {
        (void)queryHeads;
        LoadData3DParamsV2<half> params;
        params.l1H = MAX_QUERY_HEADS / NZ_INNER;
        params.l1W = NZ_INNER;
        params.channelSize = headDim_;
        params.padList[0] = 0;
        params.padList[1] = 0;
        params.padList[2] = 0;
        params.padList[3] = 255;
        params.mExtension = MAX_QUERY_HEADS;
        params.kExtension = headDim_;
        params.mStartPt = 0;
        params.kStartPt = 0;
        params.strideW = 1;
        params.strideH = 1;
        params.filterW = 1;
        params.filterSizeW = false;
        params.filterH = 1;
        params.filterSizeH = false;
        params.dilationFilterW = 1;
        params.dilationFilterH = 1;
        params.enTranspose = 0;
        params.fMatrixCtrl = 0;
        LoadData<half, LOAD3DV2_CONFIG>(aL0Buf_.Get<half>(), queryL1Buf_.Get<half>(), params);
    }

    __aicore__ inline void GatherKvTilePair(int64_t tokenRow, int64_t request, int64_t kvHead,
                                            int64_t tileStart, int64_t tileTokens, int64_t groupCount,
                                            int64_t tailStart, int64_t tailCount, bool densePrefix,
                                            const LocalTensor<int32_t> &localGroups,
                                            const LocalTensor<int32_t> &localBlockTable)
    {
        LocalTensor<half> keyGather = kvGatherBuf_.Get<half>();
        LocalTensor<half> valueGather = kvTransposeBuf_.Get<half>();
        const DataCopyParams groupCopy{static_cast<uint16_t>(headDimBlocks_), QSA_COMPRESS_RATIO,
                                       static_cast<uint16_t>(cacheBlockSize_ - QSA_COMPRESS_RATIO),
                                       static_cast<uint16_t>(TOKEN_TILE - QSA_COMPRESS_RATIO)};
        const DataCopyParams tokenCopy{static_cast<uint16_t>(headDimBlocks_), 1,
                                       static_cast<uint16_t>(cacheBlockSize_ - 1),
                                       static_cast<uint16_t>(TOKEN_TILE - 1)};
        int64_t localToken = 0;
        int64_t selectedToken = tileStart;
        while (localToken < tileTokens && selectedToken < groupCount * QSA_COMPRESS_RATIO) {
            const int64_t groupRank = selectedToken / QSA_COMPRESS_RATIO;
            const int64_t inGroup = selectedToken % QSA_COMPRESS_RATIO;
            const int64_t group = GroupAt(tokenRow, groupRank, densePrefix, localGroups);
            const int64_t token = group * QSA_COMPRESS_RATIO + inGroup;
            const int64_t cacheOffset = CacheOffset(request, token, kvHead, localBlockTable);
            if (inGroup == 0 && tileTokens - localToken >= QSA_COMPRESS_RATIO) {
                DataCopy(keyGather[localToken * NZ_INNER], keyCacheGm_[cacheOffset], groupCopy);
                DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[cacheOffset], groupCopy);
                localToken += QSA_COMPRESS_RATIO;
                selectedToken += QSA_COMPRESS_RATIO;
            } else {
                DataCopy(keyGather[localToken * NZ_INNER], keyCacheGm_[cacheOffset], tokenCopy);
                DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[cacheOffset], tokenCopy);
                ++localToken;
                ++selectedToken;
            }
        }
        while (localToken < tileTokens) {
            const int64_t tailOffset = selectedToken - groupCount * QSA_COMPRESS_RATIO;
            const int64_t token = tailOffset < tailCount ? tailStart + tailOffset : 0;
            const int64_t cacheOffset = CacheOffset(request, token, kvHead, localBlockTable);
            DataCopy(keyGather[localToken * NZ_INNER], keyCacheGm_[cacheOffset], tokenCopy);
            DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[cacheOffset], tokenCopy);
            ++localToken;
            ++selectedToken;
        }
        const int64_t fallback = CacheOffset(request, 0, kvHead, localBlockTable);
        while (localToken < TOKEN_TILE) {
            DataCopy(keyGather[localToken * NZ_INNER], keyCacheGm_[fallback], tokenCopy);
            DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[fallback], tokenCopy);
            ++localToken;
        }
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
    }

    __aicore__ inline void CopyGatheredKvToL1(const LocalTensor<half> &gathered)
    {
        DataCopy(kvL1Buf_.Get<half>(), gathered, TOKEN_TILE * headDim_);
        SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void LoadKvToL0(int64_t tileWidth)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = headDimBlocks_ * (tileWidth / NZ_INNER);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;
        LoadData(bL0Buf_.Get<half>(), kvL1Buf_.Get<half>(), params);
    }

    __aicore__ inline void CopyCubeToUb(LocalTensor<float> destination, int64_t nBlocks)
    {
        SetFlag<HardEvent::M_V>(EVENT_ID2);
        WaitFlag<HardEvent::M_V>(EVENT_ID2);
        DataCopyParams params{static_cast<uint16_t>(nBlocks), 1, 0, 0};
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(destination, cL0Buf_.Get<float>(), params, enhanced);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeScores(int64_t queryHeads, int64_t tileWidth)
    {
        (void)queryHeads;
        LoadKvToL0(tileWidth);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
        MmadParams params;
        params.m = MAX_QUERY_HEADS;
        params.n = tileWidth;
        params.k = headDim_;
        params.cmatrixInitVal = true;
        Mmad(cL0Buf_.Get<float>(), aL0Buf_.Get<half>(), bL0Buf_.Get<half>(), params);
        CopyCubeToUb(scoreBuf_.Get<float>(), tileWidth / NZ_INNER);
        Muls(scoreBuf_.Get<float>(), scoreBuf_.Get<float>(), scale_, MAX_QUERY_HEADS * tileWidth);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ReduceMaximum(LocalTensor<float> values, LocalTensor<float> scratch)
    {
        constexpr int64_t FP32_REPEAT_ELEMENTS = 64;
        constexpr int64_t FP32_REPEAT_BLOCKS = FP32_REPEAT_ELEMENTS / INT32_ALIGNMENT;
        // dav_m200 emits value/index pairs even for ORDER_ONLY_VALUE.  Request
        // the physical layout explicitly and consume its first value lane.
        WholeReduceMax(scratch, values, FP32_REPEAT_ELEMENTS, 1, 1, 1, FP32_REPEAT_BLOCKS,
                       ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        return scratch.GetValue(0);
    }

    __aicore__ inline float ReduceSum(LocalTensor<float> values, LocalTensor<float> scratch)
    {
        constexpr int64_t FP32_REPEAT_ELEMENTS = 64;
        constexpr int64_t FP32_REPEAT_BLOCKS = FP32_REPEAT_ELEMENTS / INT32_ALIGNMENT;
        WholeReduceSum(scratch, values, FP32_REPEAT_ELEMENTS, 1, 1, 1, FP32_REPEAT_BLOCKS);
        PipeBarrier<PIPE_V>();
        return scratch.GetValue(0);
    }

    __aicore__ inline float ScalarExp(float value, LocalTensor<float> scratch)
    {
        Duplicate(scratch, value, INT32_ALIGNMENT);
        PipeBarrier<PIPE_V>();
        Exp(scratch, scratch, INT32_ALIGNMENT);
        PipeBarrier<PIPE_V>();
        return scratch.GetValue(0);
    }

    __aicore__ inline void SoftmaxTile(int64_t queryHeads, int64_t tileTokens, int64_t tileWidth,
                                      float *rowMax, float *rowSum, float *previousWeight)
    {
        LocalTensor<float> scores = scoreBuf_.Get<float>();
        LocalTensor<float> softmaxRows = softmaxBuf_.Get<float>();
        LocalTensor<half> probabilityRows = probabilityBuf_.Get<half>();
        LocalTensor<half> probabilities = scoreBuf_.Get<half>();
        LocalTensor<float> scratch = reduceBuf_.Get<float>();
        const int64_t nBlocks = tileWidth / NZ_INNER;
        // Cube stores C as [N/16, M, 16].  Repack all valid rows in eight
        // strided vector operations so softmax can process 128 contiguous
        // scores per head instead of issuing one 16-lane operation per block.
        const UnaryRepeatParams scoresToRows{
            1, 1, static_cast<uint8_t>(TOKEN_TILE / INT32_ALIGNMENT),
            static_cast<uint8_t>(NZ_INNER / INT32_ALIGNMENT)};
        for (int64_t block = 0; block < nBlocks; ++block) {
            Adds(softmaxRows[block * NZ_INNER], scores[block * MAX_QUERY_HEADS * NZ_INNER], 0.0f,
                 NZ_INNER, queryHeads, scoresToRows);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t head = 0; head < queryHeads; ++head) {
            const int64_t rowOffset = head * TOKEN_TILE;
            for (int64_t lane = tileTokens; lane < tileWidth; ++lane) {
                softmaxRows.SetValue(rowOffset + lane, -3.402823466e38f);
            }
            PipeBarrier<PIPE_V>();
            const float tileMax = ReduceMaximum(softmaxRows[rowOffset], scratch);
            const float newMax = rowSum[head] == 0.0f || tileMax > rowMax[head] ? tileMax : rowMax[head];
            previousWeight[head] = rowSum[head] == 0.0f ? 0.0f : ScalarExp(rowMax[head] - newMax, scratch);
            Adds(softmaxRows[rowOffset], softmaxRows[rowOffset], -newMax, tileWidth);
            PipeBarrier<PIPE_V>();
            rowMax[head] = newMax;
        }
        // All active rows are contiguous in UB. One vector exponential avoids
        // repeating pipeline setup once per query head for every 64-token
        // tile while preserving the per-row FP32 max and reduction order.
        Exp(softmaxRows, softmaxRows, queryHeads * tileWidth);
        PipeBarrier<PIPE_V>();
        for (int64_t head = 0; head < queryHeads; ++head) {
            const int64_t rowOffset = head * TOKEN_TILE;
            for (int64_t lane = tileTokens; lane < tileWidth; ++lane) {
                softmaxRows.SetValue(rowOffset + lane, 0.0f);
            }
            PipeBarrier<PIPE_V>();
            const float tileSum = ReduceSum(softmaxRows[rowOffset], scratch);
            rowSum[head] = rowSum[head] * previousWeight[head] + tileSum;
        }
        Cast(probabilityRows, softmaxRows, RoundMode::CAST_NONE, queryHeads * tileWidth);
        PipeBarrier<PIPE_V>();
        const UnaryRepeatParams rowsToProbabilityNz{
            1, 1, 1, static_cast<uint8_t>(TOKEN_TILE * sizeof(half) / 32)};
        for (int64_t block = 0; block < nBlocks; ++block) {
            Adds(probabilities[block * MAX_QUERY_HEADS * NZ_INNER], probabilityRows[block * NZ_INNER],
                 static_cast<half>(0), NZ_INNER, queryHeads, rowsToProbabilityNz);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeValues(int64_t queryHeads, int64_t tileWidth)
    {
        (void)queryHeads;
        LocalTensor<half> probabilityL1 = probabilityL1Buf_.Get<half>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
        DataCopy(probabilityL1, scoreBuf_.Get<half>(), MAX_QUERY_HEADS * tileWidth);
        SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID4);

        LoadData3DParamsV2<half> probabilityParams;
        probabilityParams.l1H = MAX_QUERY_HEADS / NZ_INNER;
        probabilityParams.l1W = NZ_INNER;
        probabilityParams.channelSize = tileWidth;
        probabilityParams.padList[0] = 0;
        probabilityParams.padList[1] = 0;
        probabilityParams.padList[2] = 0;
        probabilityParams.padList[3] = 255;
        probabilityParams.mExtension = MAX_QUERY_HEADS;
        probabilityParams.kExtension = tileWidth;
        probabilityParams.mStartPt = 0;
        probabilityParams.kStartPt = 0;
        probabilityParams.strideW = 1;
        probabilityParams.strideH = 1;
        probabilityParams.filterW = 1;
        probabilityParams.filterSizeW = false;
        probabilityParams.filterH = 1;
        probabilityParams.filterSizeH = false;
        probabilityParams.dilationFilterW = 1;
        probabilityParams.dilationFilterH = 1;
        probabilityParams.enTranspose = 0;
        probabilityParams.fMatrixCtrl = 0;
        LocalTensor<half> probabilityL0 = aL0Buf_.Get<half>()[QUERY_L0_ELEMENTS];
        LoadData<half, LOAD3DV2_CONFIG>(probabilityL0, probabilityL1, probabilityParams);

        // The gathered value tile is NZ storage for logical [K, N].  B2 needs
        // the transposed Load3D path used by the production sparse-attention
        // matmul service; a plain LoadData2D changes the K accumulation order
        // enough to miss the serving accuracy contract on partial head tiles.
        LoadData3DParamsV2<half> valueParams;
        valueParams.l1H = tileWidth / NZ_INNER;
        valueParams.l1W = NZ_INNER;
        valueParams.channelSize = headDim_;
        valueParams.padList[0] = 0;
        valueParams.padList[1] = 0;
        valueParams.padList[2] = 0;
        valueParams.padList[3] = 255;
        valueParams.mExtension = tileWidth;
        valueParams.kExtension = headDim_;
        valueParams.mStartPt = 0;
        valueParams.kStartPt = 0;
        valueParams.strideW = 1;
        valueParams.strideH = 1;
        valueParams.filterW = 1;
        valueParams.filterSizeW = false;
        valueParams.filterH = 1;
        valueParams.filterSizeH = false;
        valueParams.dilationFilterW = 1;
        valueParams.dilationFilterH = 1;
        valueParams.enTranspose = 1;
        valueParams.fMatrixCtrl = 0;
        LoadData<half, LOAD3DV2_CONFIG>(bL0Buf_.Get<half>(), kvL1Buf_.Get<half>(), valueParams);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID5);
        MmadParams params;
        params.m = MAX_QUERY_HEADS;
        params.n = headDim_;
        params.k = tileWidth;
        params.cmatrixInitVal = true;
        Mmad(cL0Buf_.Get<float>(), probabilityL0, bL0Buf_.Get<half>(), params);
        CopyCubeToUb(contributionBuf_.Get<float>(), headDimBlocks_);
    }

    __aicore__ inline void AccumulateValues(int64_t queryHeads, const float *previousWeight)
    {
        LocalTensor<float> contribution = contributionBuf_.Get<float>();
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        for (int64_t dimBlock = 0; dimBlock < headDimBlocks_; ++dimBlock) {
            const int64_t blockOffset = dimBlock * MAX_QUERY_HEADS * NZ_INNER;
            for (int64_t head = 0; head < queryHeads; ++head) {
                const int64_t offset = blockOffset + head * NZ_INNER;
                if (previousWeight[head] == 0.0f) {
                    Adds(accumulator[offset], contribution[offset], 0.0f, NZ_INNER);
                } else {
                    Muls(accumulator[offset], accumulator[offset], previousWeight[head], NZ_INNER);
                    PipeBarrier<PIPE_V>();
                    Add(accumulator[offset], accumulator[offset], contribution[offset], NZ_INNER);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOutput(int64_t queryOffset, int64_t queryHeads, const float *rowSum)
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<half> output = outputBuf_.Get<half>();
        for (int64_t head = 0; head < queryHeads; ++head) {
            const float inverse = rowSum[head] > 0.0f ? 1.0f / rowSum[head] : 0.0f;
            for (int64_t dimBlock = 0; dimBlock < headDimBlocks_; ++dimBlock) {
                const int64_t source = dimBlock * MAX_QUERY_HEADS * NZ_INNER + head * NZ_INNER;
                const int64_t destination = head * headDim_ + dimBlock * NZ_INNER;
                Muls(accumulator[source], accumulator[source], inverse, NZ_INNER);
                PipeBarrier<PIPE_V>();
                Cast(output[destination], accumulator[source], RoundMode::CAST_NONE, NZ_INNER);
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
        DataCopy(outputGm_[queryOffset], output, queryHeads * headDim_);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }

    __aicore__ inline void ComputeTask(int64_t task, const LocalTensor<int32_t> &localBlockTable)
    {
        const int64_t tokenRow = task / (numKvHeads_ * taskTilesPerKvHead_);
        const int64_t kvHeadTask = task % (numKvHeads_ * taskTilesPerKvHead_);
        const int64_t kvHead = kvHeadTask / taskTilesPerKvHead_;
        const int64_t headStart = (kvHeadTask % taskTilesPerKvHead_) * headsPerTask_;
        const int64_t queryHeads = totalQueryHeadsPerKvHead_ - headStart < headsPerTask_
                                       ? totalQueryHeadsPerKvHead_ - headStart : headsPerTask_;
        const int64_t request = RequestForToken(tokenRow);
        const int64_t queryOffset =
            (tokenRow * numQueryHeads_ + kvHead * totalQueryHeadsPerKvHead_ + headStart) * headDim_;
        LoadQuery(queryOffset, queryHeads);
        // Q is invariant across all selected-token tiles in this task. Keep
        // its A2 representation resident while PV uses a disjoint A2 region.
        LoadQueryToL0(queryHeads);

        const int64_t encodedGroupCount = groupCountsGm_.GetValue(tokenRow);
        const int64_t encodedTailCount = tailCountsGm_.GetValue(tokenRow);
        const bool denseTokenPrefix = encodedTailCount < 0;
        const bool densePrefix = denseTokenPrefix || encodedGroupCount < 0;
        const int64_t groupCount = denseTokenPrefix
                                       ? encodedGroupCount / QSA_COMPRESS_RATIO
                                       : (encodedGroupCount < 0 ? -encodedGroupCount : encodedGroupCount);
        const int64_t tailStart = denseTokenPrefix ? groupCount * QSA_COMPRESS_RATIO
                                                    : tailStartsGm_.GetValue(tokenRow);
        const int64_t tailCount = denseTokenPrefix ? encodedGroupCount - tailStart : encodedTailCount;
        const int64_t selectedTokens = groupCount * QSA_COMPRESS_RATIO + tailCount;
        LocalTensor<int32_t> localGroups;
        if (!densePrefix && useLocalGroups_) {
            localGroups = groupBuf_.Get<int32_t>();
            DataCopy(localGroups, groupIndicesGm_[tokenRow * selectedGroupsWidth_], selectedGroupsWidth_);
            PipeBarrier<PIPE_ALL>();
        }

        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        Duplicate(accumulator, 0.0f, MAX_QUERY_HEADS * headDim_);
        PipeBarrier<PIPE_V>();
        float rowMax[MAX_QUERY_HEADS];
        float rowSum[MAX_QUERY_HEADS];
        float previousWeight[MAX_QUERY_HEADS];
        for (int64_t head = 0; head < queryHeads; ++head) {
            rowMax[head] = -3.402823466e38f;
            rowSum[head] = 0.0f;
        }

        for (int64_t tileStart = 0; tileStart < selectedTokens; tileStart += TOKEN_TILE) {
            const int64_t tileTokens = selectedTokens - tileStart < TOKEN_TILE ? selectedTokens - tileStart
                                                                               : TOKEN_TILE;
            // Keep the L1 NZ stride fixed at 64 rows. The last tile masks its
            // unused score lanes and supplies zero probabilities to PV.
            const int64_t tileWidth = TOKEN_TILE;
            GatherKvTilePair(tokenRow, request, kvHead, tileStart, tileTokens, groupCount, tailStart,
                             tailCount, densePrefix, localGroups, localBlockTable);
            CopyGatheredKvToL1(kvGatherBuf_.Get<half>());
            ComputeScores(queryHeads, tileWidth);
            SoftmaxTile(queryHeads, tileTokens, tileWidth, rowMax, rowSum, previousWeight);
            CopyGatheredKvToL1(kvTransposeBuf_.Get<half>());
            ComputeValues(queryHeads, tileWidth);
            AccumulateValues(queryHeads, previousWeight);
        }
        StoreOutput(queryOffset, queryHeads, rowSum);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::A1> queryL1Buf_;
    TBuf<TPosition::A1> probabilityL1Buf_;
    TBuf<TPosition::B1> kvL1Buf_;
    TBuf<TPosition::A2> aL0Buf_;
    TBuf<TPosition::B2> bL0Buf_;
    TBuf<TPosition::CO1> cL0Buf_;
    TBuf<TPosition::VECCALC> kvGatherBuf_;
    TBuf<TPosition::VECCALC> kvTransposeBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> softmaxBuf_;
    TBuf<TPosition::VECCALC> probabilityBuf_;
    TBuf<TPosition::VECCALC> contributionBuf_;
    TBuf<TPosition::VECCALC> accumulatorBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> groupBuf_;
    TBuf<TPosition::VECCALC> blockTableBuf_;
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
    int64_t headDimBlocks_ = 0;
    int64_t cacheBlockSize_ = 0;
    int64_t cacheHeadDimBlocks_ = 0;
    int64_t maxBlocksPerSequence_ = 0;
    int64_t selectedGroupsWidth_ = 0;
    int64_t numRequests_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t taskCount_ = 0;
    float scale_ = 1.0f;
    bool useLocalGroups_ = false;
    bool useLocalBlockTable_ = false;
};

}  // namespace NsQsaCubeSparseAttention

#endif
