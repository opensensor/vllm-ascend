#ifndef SIDECARD_MHC_PRE_V310_TILING_H
#define SIDECARD_MHC_PRE_V310_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(SidecardMhcPreV310TilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, blocks);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, hcEps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SidecardMhcPreV310, SidecardMhcPreV310TilingData)
}  // namespace optiling
#endif
