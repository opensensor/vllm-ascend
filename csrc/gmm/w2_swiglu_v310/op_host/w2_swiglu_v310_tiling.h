// SPDX-License-Identifier: Apache-2.0
#ifndef W2_SWIGLU_TILING_H
#define W2_SWIGLU_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(W2SwigluTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, width);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(W2SwigluV310, W2SwigluTilingData)
}  // namespace optiling
#endif
