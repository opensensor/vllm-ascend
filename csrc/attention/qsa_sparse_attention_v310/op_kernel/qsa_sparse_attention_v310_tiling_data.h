#ifndef QSA_SPARSE_ATTENTION_V310_TILING_DATA_H
#define QSA_SPARSE_ATTENTION_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct QsaSparseAttentionV310TilingData {
    int64_t numTokens;
    int64_t numQueryHeads;
    int64_t numKvHeads;
    int64_t headDim;
    int64_t cacheBlockSize;
    int64_t cacheHeadDimBlocks;
    int64_t maxBlocksPerSequence;
    int64_t selectedGroupsWidth;
    int64_t numRequests;
    int64_t tasksPerCore;
    int64_t taskCount;
    int64_t scaleQ24;
};

#endif
