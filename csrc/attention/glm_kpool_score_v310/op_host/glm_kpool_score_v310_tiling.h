// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(GlmKpoolScoreTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, pools);
TILING_DATA_FIELD_DEF(int64_t, requests);
TILING_DATA_FIELD_DEF(int64_t, columns);
TILING_DATA_FIELD_DEF(int64_t, blocks);
TILING_DATA_FIELD_DEF(int64_t, block_rows);
TILING_DATA_FIELD_DEF(int64_t, block_stride);
TILING_DATA_FIELD_DEF(int64_t, row_stride);
TILING_DATA_FIELD_DEF(int64_t, cache_offset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GlmKpoolScoreV310, GlmKpoolScoreTilingData)
}
