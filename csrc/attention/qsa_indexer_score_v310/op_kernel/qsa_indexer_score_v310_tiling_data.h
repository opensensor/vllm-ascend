#ifndef QSA_INDEXER_SCORE_V310_TILING_DATA_H
#define QSA_INDEXER_SCORE_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct QsaIndexerScoreV310TilingData {
    int64_t numTokens;
    int64_t numHeads;
    int64_t headDim;
    int64_t cacheRowsPerBlock;
    int64_t groupsPerBlock;
    int64_t maxBlocksPerSequence;
    int64_t maxGroupsPerSequence;
    int64_t numRequests;
    int64_t tasksPerCore;
    int64_t taskCount;
};

#endif
