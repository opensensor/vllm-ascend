// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef QSA_GATHER_VALUE_NZ_V310_H
#define QSA_GATHER_VALUE_NZ_V310_H

#include "kernel_operator.h"
#include "qsa_gather_value_nz_v310_tiling_data.h"

namespace NsQsaGatherValueNz {

using namespace AscendC;

constexpr int64_t NZ_INNER = 16;
constexpr int64_t COMPRESS_RATIO = 4;
constexpr int64_t GROUP_ELEMENTS = COMPRESS_RATIO * NZ_INNER;
constexpr int64_t TILE_ELEMENTS = NZ_INNER * NZ_INNER;
// Stage four NZ token blocks (16 compression groups) before each barrier.
// The value layout can flush those blocks with one strided DMA per head
// dimension block; the transposed key layout flushes them one NZ block at a
// time after transposition.
constexpr int64_t TOKEN_BLOCKS_PER_TILE = 4;
constexpr int64_t TILE_BUFFER_ELEMENTS = TOKEN_BLOCKS_PER_TILE * TILE_ELEMENTS;
constexpr int64_t GROUP_INDEX_ALIGNMENT = 8;
constexpr int64_t BLOCK_TABLE_ALIGNMENT = 8;
// 128K tokens at a 64-token page size fit in 8 KiB of local table storage.
constexpr int64_t MAX_LOCAL_BLOCK_TABLE_ENTRIES = 2048;

class QsaGatherValueNzV310 {
public:
    __aicore__ inline void Init(GM_ADDR valueCache, GM_ADDR groupIndices, GM_ADDR groupCounts,
                                GM_ADDR tailStarts, GM_ADDR tailCounts, GM_ADDR blockTable,
                                GM_ADDR valueNz, const QsaGatherValueNzV310TilingData *tiling,
                                TPipe *pipe)
    {
        numKvHeads_ = tiling->numKvHeads;
        headDimBlocks_ = tiling->headDimBlocks;
        cacheHeadDimBlocks_ = tiling->cacheHeadDimBlocks;
        cacheBlockSize_ = tiling->cacheBlockSize;
        selectedGroupsWidth_ = tiling->selectedGroupsWidth;
        outputTokenBlocks_ = tiling->outputTokenBlocks;
        maxBlocksPerSequence_ = tiling->maxBlocksPerSequence;
        taskCount_ = tiling->taskCount;
        tasksPerCore_ = tiling->tasksPerCore;
        transposeOutput_ = tiling->transposeOutput != 0;
        useLocalBlockTable_ = maxBlocksPerSequence_ > 0 && maxBlocksPerSequence_ % BLOCK_TABLE_ALIGNMENT == 0 &&
                              maxBlocksPerSequence_ <= MAX_LOCAL_BLOCK_TABLE_ENTRIES;
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(valueCache));
        groupIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupIndices));
        groupCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupCounts));
        tailStartsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tailStarts));
        tailCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tailCounts));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        valueNzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(valueNz));
        // Four 16-token NZ blocks across every head-dimension block. Grouped
        // strided DMA fills all dimensions with one transfer per group.
        pipe->InitBuffer(tileBuf_, headDimBlocks_ * TILE_BUFFER_ELEMENTS * sizeof(half));
        if (transposeOutput_) {
            pipe->InitBuffer(transposedTileBuf_, headDimBlocks_ * TILE_BUFFER_ELEMENTS * sizeof(half));
        }
        if (selectedGroupsWidth_ % GROUP_INDEX_ALIGNMENT == 0) {
            pipe->InitBuffer(groupBuf_, selectedGroupsWidth_ * sizeof(int32_t));
        }
        if (useLocalBlockTable_) {
            pipe->InitBuffer(blockTableBuf_, maxBlocksPerSequence_ * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> localBlockTable;
        if (useLocalBlockTable_) {
            // Every selected group otherwise performs a scalar GM page lookup
            // once per head-dimension block. Share one sequential copy across
            // all tasks assigned to this AI Vector core.
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
            CopyTask(task, localBlockTable);
        }
    }

private:
    __aicore__ inline int64_t CacheOffset(int64_t token, int64_t kvHead, int64_t dimBlock,
                                         const LocalTensor<int32_t>& localBlockTable) const
    {
        const int64_t logicalBlock = token / cacheBlockSize_;
        const int64_t physicalBlock = useLocalBlockTable_ ? localBlockTable.GetValue(logicalBlock)
                                                         : blockTableGm_.GetValue(logicalBlock);
        return ((physicalBlock * cacheHeadDimBlocks_ + kvHead * headDimBlocks_ + dimBlock) * cacheBlockSize_
                + token % cacheBlockSize_) * NZ_INNER;
    }

    __aicore__ inline void CopyTask(int64_t task, const LocalTensor<int32_t>& localBlockTable)
    {
        const int64_t kvHead = task % numKvHeads_;
        const int64_t tokenRow = task / numKvHeads_;
        const int64_t groupCount = groupCountsGm_.GetValue(tokenRow);
        const int64_t tailStart = tailStartsGm_.GetValue(tokenRow);
        const int64_t tailCount = tailCountsGm_.GetValue(tokenRow);
        LocalTensor<half> tile = tileBuf_.Get<half>();
        LocalTensor<half> transposedTile;
        if (transposeOutput_) {
            transposedTile = transposedTileBuf_.Get<half>();
        }
        LocalTensor<int32_t> groupIndices;
        if (selectedGroupsWidth_ % GROUP_INDEX_ALIGNMENT == 0) {
            groupIndices = groupBuf_.Get<int32_t>();
            DataCopy(groupIndices, groupIndicesGm_[tokenRow * selectedGroupsWidth_], selectedGroupsWidth_);
            PipeBarrier<PIPE_ALL>();
        }

        const DataCopyParams groupCopy{static_cast<uint16_t>(headDimBlocks_),
                                       static_cast<uint16_t>(GROUP_ELEMENTS / NZ_INNER),
                                       static_cast<uint16_t>(cacheBlockSize_ - COMPRESS_RATIO),
                                       static_cast<uint16_t>((TILE_BUFFER_ELEMENTS - GROUP_ELEMENTS) / NZ_INNER)};
        const DataCopyParams tailCopy{static_cast<uint16_t>(headDimBlocks_), 1,
                                      static_cast<uint16_t>(cacheBlockSize_ - 1),
                                      static_cast<uint16_t>((TILE_BUFFER_ELEMENTS - NZ_INNER) / NZ_INNER)};
        const int64_t fallbackOffset = CacheOffset(0, kvHead, 0, localBlockTable);
        for (int64_t block = 0; block < outputTokenBlocks_; block += TOKEN_BLOCKS_PER_TILE) {
            const int64_t blocksThisTile = outputTokenBlocks_ - block < TOKEN_BLOCKS_PER_TILE
                                               ? outputTokenBlocks_ - block : TOKEN_BLOCKS_PER_TILE;
            for (int64_t lane = 0; lane < blocksThisTile * NZ_INNER / COMPRESS_RATIO; ++lane) {
                const int64_t groupRank = block * (NZ_INNER / COMPRESS_RATIO) + lane;
                const int64_t tileOffset = lane * GROUP_ELEMENTS;
                if (groupRank < selectedGroupsWidth_ && groupRank < groupCount) {
                    const int64_t group = selectedGroupsWidth_ % GROUP_INDEX_ALIGNMENT == 0
                                              ? groupIndices.GetValue(groupRank)
                                              : groupIndicesGm_.GetValue(tokenRow * selectedGroupsWidth_ + groupRank);
                    const int64_t cacheOffset = CacheOffset(group * COMPRESS_RATIO, kvHead, 0, localBlockTable);
                    DataCopy(tile[tileOffset], valueCacheGm_[cacheOffset], groupCopy);
                } else if (groupRank == selectedGroupsWidth_ && tailCount > 0) {
                    for (int64_t tail = 0; tail < COMPRESS_RATIO; ++tail) {
                        const int64_t token = tail < tailCount ? tailStart + tail : 0;
                        const int64_t cacheOffset = CacheOffset(token, kvHead, 0, localBlockTable);
                        DataCopy(tile[tileOffset + tail * NZ_INNER], valueCacheGm_[cacheOffset], tailCopy);
                    }
                } else {
                    DataCopy(tile[tileOffset], valueCacheGm_[fallbackOffset], groupCopy);
                }
            }
            PipeBarrier<PIPE_ALL>();
            if (transposeOutput_) {
                // For score GEMM, output is logical [T,H,D,S] in NZ. Transpose
                // each 16x16 tile in UB, then write adjacent D blocks once per
                // staged token block.
                for (int64_t dimBlock = 0; dimBlock < headDimBlocks_; ++dimBlock) {
                    for (int64_t tokenBlock = 0; tokenBlock < blocksThisTile; ++tokenBlock) {
                        Transpose(
                            transposedTile[(tokenBlock * headDimBlocks_ + dimBlock) * TILE_ELEMENTS],
                            tile[dimBlock * TILE_BUFFER_ELEMENTS + tokenBlock * TILE_ELEMENTS]);
                    }
                }
                PipeBarrier<PIPE_ALL>();
                for (int64_t tokenBlock = 0; tokenBlock < blocksThisTile; ++tokenBlock) {
                    const int64_t outputOffset =
                        ((tokenRow * numKvHeads_ + kvHead) * outputTokenBlocks_ * headDimBlocks_
                         + (block + tokenBlock) * headDimBlocks_) * TILE_ELEMENTS;
                    DataCopy(valueNzGm_[outputOffset],
                             transposedTile[tokenBlock * headDimBlocks_ * TILE_ELEMENTS],
                             headDimBlocks_ * TILE_ELEMENTS);
                }
            } else {
                // Value output is logical [T,H,S,D] in NZ storage.
                const DataCopyParams outputCopy{
                    static_cast<uint16_t>(headDimBlocks_),
                    static_cast<uint16_t>(blocksThisTile * TILE_ELEMENTS / NZ_INNER),
                    static_cast<uint16_t>((TOKEN_BLOCKS_PER_TILE - blocksThisTile) * TILE_ELEMENTS / NZ_INNER),
                    static_cast<uint16_t>((outputTokenBlocks_ - blocksThisTile) * TILE_ELEMENTS / NZ_INNER)};
                const int64_t outputOffset =
                    ((tokenRow * numKvHeads_ + kvHead) * headDimBlocks_ * outputTokenBlocks_ + block)
                    * TILE_ELEMENTS;
                DataCopy(valueNzGm_[outputOffset], tile, outputCopy);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    TBuf<TPosition::VECCALC> tileBuf_;
    TBuf<TPosition::VECCALC> transposedTileBuf_;
    TBuf<TPosition::VECCALC> groupBuf_;
    TBuf<TPosition::VECCALC> blockTableBuf_;
    GlobalTensor<half> valueCacheGm_;
    GlobalTensor<int32_t> groupIndicesGm_;
    GlobalTensor<int32_t> groupCountsGm_;
    GlobalTensor<int32_t> tailStartsGm_;
    GlobalTensor<int32_t> tailCountsGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<half> valueNzGm_;
    int64_t numKvHeads_ = 0;
    int64_t headDimBlocks_ = 0;
    int64_t cacheHeadDimBlocks_ = 0;
    int64_t cacheBlockSize_ = 0;
    int64_t selectedGroupsWidth_ = 0;
    int64_t outputTokenBlocks_ = 0;
    int64_t maxBlocksPerSequence_ = 0;
    int64_t taskCount_ = 0;
    int64_t tasksPerCore_ = 0;
    bool useLocalBlockTable_ = false;
    bool transposeOutput_ = false;
};

}  // namespace NsQsaGatherValueNz

#endif
