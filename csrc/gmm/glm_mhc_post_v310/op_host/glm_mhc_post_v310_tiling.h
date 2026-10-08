// SPDX-License-Identifier: Apache-2.0
#ifndef GLM_MHC_POST_TILING_H
#define GLM_MHC_POST_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(GlmMhcPostTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, width);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GlmMhcPostV310, GlmMhcPostTilingData)
}  // namespace optiling
#endif
