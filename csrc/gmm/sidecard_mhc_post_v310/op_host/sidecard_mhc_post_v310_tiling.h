#ifndef SIDECARD_MHC_POST_V310_TILING_H
#define SIDECARD_MHC_POST_V310_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(SidecardMhcPostV310TilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, blocks);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SidecardMhcPostV310, SidecardMhcPostV310TilingData)
}  // namespace optiling
#endif
