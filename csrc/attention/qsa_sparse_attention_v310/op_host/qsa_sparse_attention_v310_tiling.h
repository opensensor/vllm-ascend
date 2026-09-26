#ifndef ASCEND_OPS_QSA_SPARSE_ATTENTION_V310_TILING_H
#define ASCEND_OPS_QSA_SPARSE_ATTENTION_V310_TILING_H

#include <cstdint>

#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QsaSparseAttentionV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numTokens);
    TILING_DATA_FIELD_DEF(int64_t, numQueryHeads);
    TILING_DATA_FIELD_DEF(int64_t, numKvHeads);
    TILING_DATA_FIELD_DEF(int64_t, headsPerTask);
    TILING_DATA_FIELD_DEF(int64_t, taskTilesPerKvHead);
    TILING_DATA_FIELD_DEF(int64_t, headDim);
    TILING_DATA_FIELD_DEF(int64_t, cacheBlockSize);
    TILING_DATA_FIELD_DEF(int64_t, cacheHeadDimBlocks);
    TILING_DATA_FIELD_DEF(int64_t, maxBlocksPerSequence);
    TILING_DATA_FIELD_DEF(int64_t, selectedGroupsWidth);
    TILING_DATA_FIELD_DEF(int64_t, numRequests);
    TILING_DATA_FIELD_DEF(int64_t, tasksPerCore);
    TILING_DATA_FIELD_DEF(int64_t, taskCount);
    TILING_DATA_FIELD_DEF(int64_t, scaleQ24);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QsaSparseAttentionV310, QsaSparseAttentionV310TilingData)

}  // namespace optiling

#endif
