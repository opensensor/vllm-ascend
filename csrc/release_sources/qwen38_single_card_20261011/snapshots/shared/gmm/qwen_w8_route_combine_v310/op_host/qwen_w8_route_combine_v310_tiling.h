// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W8_ROUTE_COMBINE_V310_TILING_H
#define QWEN_W8_ROUTE_COMBINE_V310_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QwenW8RouteCombineTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(int64_t, topK);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QwenW8RouteCombineV310, QwenW8RouteCombineTilingData)
}  // namespace optiling
#endif
