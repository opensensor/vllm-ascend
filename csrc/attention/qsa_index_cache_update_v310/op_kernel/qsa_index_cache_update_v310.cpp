#include "qsa_index_cache_update_v310.h"

extern "C" __global__ __aicore__ void qsa_index_cache_update_v310(
    GM_ADDR compressedKeyCache, GM_ADDR indexKeys, GM_ADDR queryStartLoc,
    GM_ADDR slotMapping, GM_ADDR keyNormWeight, GM_ADDR ropeCos, GM_ADDR ropeSin,
    GM_ADDR compressedKeyCacheOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(QsaIndexCacheUpdateV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(QsaIndexCacheUpdateV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsQsaIndexCacheUpdate::QsaIndexCacheUpdateV310 op;
    op.Init(compressedKeyCache, indexKeys, queryStartLoc, slotMapping,
            keyNormWeight, ropeCos, ropeSin,
            compressedKeyCacheOut, &tilingData, &pipe);
    op.Process();
}
