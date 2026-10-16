#ifndef SIDECARD_KDA_V310_TILING_H
#define SIDECARD_KDA_V310_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(SidecardKdaV310TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, heads);
    TILING_DATA_FIELD_DEF(uint32_t, blocks);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SidecardKdaV310, SidecardKdaV310TilingData)
}  // namespace optiling
#endif
