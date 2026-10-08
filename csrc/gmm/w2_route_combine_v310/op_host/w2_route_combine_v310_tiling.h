// SPDX-License-Identifier: Apache-2.0
#ifndef W2_ROUTE_COMBINE_TILING_H
#define W2_ROUTE_COMBINE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(W2RouteCombineTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokens);
TILING_DATA_FIELD_DEF(int64_t, hidden);
TILING_DATA_FIELD_DEF(int64_t, top_k);
TILING_DATA_FIELD_DEF(int64_t, experts);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(W2RouteCombineV310, W2RouteCombineTilingData)
}  // namespace optiling
#endif
