// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef QSA_GATHER_VALUE_NZ_V310_TILING_DATA_H
#define QSA_GATHER_VALUE_NZ_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct QsaGatherValueNzV310TilingData {
    int64_t numTokens;
    int64_t numKvHeads;
    int64_t headDimBlocks;
    int64_t cacheHeadDimBlocks;
    int64_t cacheBlockSize;
    int64_t selectedGroupsWidth;
    int64_t outputTokenBlocks;
    int64_t maxBlocksPerSequence;
    int64_t taskCount;
    int64_t tasksPerCore;
    int64_t transposeOutput;
};

#endif
