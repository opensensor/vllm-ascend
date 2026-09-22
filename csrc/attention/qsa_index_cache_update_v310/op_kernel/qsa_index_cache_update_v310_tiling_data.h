#ifndef QSA_INDEX_CACHE_UPDATE_V310_TILING_DATA_H
#define QSA_INDEX_CACHE_UPDATE_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct QsaIndexCacheUpdateV310TilingData {
    int64_t numTokens;
    int64_t headDim;
    int64_t cacheRowsPerBlock;
    int64_t groupsPerBlock;
    int64_t blockSize;
    int64_t rotaryDim;
    int64_t numRequests;
    int64_t requestsPerCore;
    float normEps;
};

#endif
