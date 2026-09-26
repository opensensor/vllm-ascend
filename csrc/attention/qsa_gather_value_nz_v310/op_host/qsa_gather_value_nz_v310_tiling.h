// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef ASCEND_OPS_QSA_GATHER_VALUE_NZ_V310_TILING_H
#define ASCEND_OPS_QSA_GATHER_VALUE_NZ_V310_TILING_H

#include <cstdint>

#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QsaGatherValueNzV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numTokens);
    TILING_DATA_FIELD_DEF(int64_t, numKvHeads);
    TILING_DATA_FIELD_DEF(int64_t, headDimBlocks);
    TILING_DATA_FIELD_DEF(int64_t, cacheHeadDimBlocks);
    TILING_DATA_FIELD_DEF(int64_t, cacheBlockSize);
    TILING_DATA_FIELD_DEF(int64_t, selectedGroupsWidth);
    TILING_DATA_FIELD_DEF(int64_t, outputTokenBlocks);
    TILING_DATA_FIELD_DEF(int64_t, maxBlocksPerSequence);
    TILING_DATA_FIELD_DEF(int64_t, taskCount);
    TILING_DATA_FIELD_DEF(int64_t, tasksPerCore);
    TILING_DATA_FIELD_DEF(int64_t, transposeOutput);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QsaGatherValueNzV310, QsaGatherValueNzV310TilingData)

}  // namespace optiling

#endif
