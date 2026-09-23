#ifndef ASCEND_OPS_QSA_INDEX_CACHE_UPDATE_V310_TILING_H
#define ASCEND_OPS_QSA_INDEX_CACHE_UPDATE_V310_TILING_H

#include <cstdint>

#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QsaIndexCacheUpdateV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numTokens);
    TILING_DATA_FIELD_DEF(int64_t, headDim);
    TILING_DATA_FIELD_DEF(int64_t, cacheRowsPerBlock);
    TILING_DATA_FIELD_DEF(int64_t, groupsPerBlock);
    TILING_DATA_FIELD_DEF(int64_t, blockSize);
    TILING_DATA_FIELD_DEF(int64_t, rotaryDim);
    TILING_DATA_FIELD_DEF(int64_t, numRequests);
    TILING_DATA_FIELD_DEF(int64_t, requestsPerCore);
    TILING_DATA_FIELD_DEF(float, normEps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QsaIndexCacheUpdateV310, QsaIndexCacheUpdateV310TilingData)

}  // namespace optiling

#endif
