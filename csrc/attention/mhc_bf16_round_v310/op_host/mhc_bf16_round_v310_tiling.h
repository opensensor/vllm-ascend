#ifndef MHC_BF16_ROUND_V310_TILING_H
#define MHC_BF16_ROUND_V310_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MhcBf16RoundV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numel);
    TILING_DATA_FIELD_DEF(uint32_t, blockCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MhcBf16RoundV310, MhcBf16RoundV310TilingData)
}  // namespace optiling

#endif
