/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gdn_gating_v310_tiling.h
 * \brief
 */
#ifndef ASCEND_OPS_GDN_GATING_V310_TILING_H
#define ASCEND_OPS_GDN_GATING_V310_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GdnGatingTilingData)
    TILING_DATA_FIELD_DEF(int64_t, numRows);      // padded T (multiple of TILE_ROWS)
    TILING_DATA_FIELD_DEF(int64_t, numHeads);     // H
    TILING_DATA_FIELD_DEF(int64_t, tileRows);    // rows per aligned DataCopy block
    TILING_DATA_FIELD_DEF(int64_t, tilesPerCore); // whole TILE_ROWS blocks per core
    TILING_DATA_FIELD_DEF(int64_t, tileCount);    // total TILE_ROWS blocks
    TILING_DATA_FIELD_DEF(float, beta);
    TILING_DATA_FIELD_DEF(float, invBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GdnGatingV310, GdnGatingTilingData)

}  // namespace optiling

#endif  // ASCEND_OPS_GDN_GATING_V310_TILING_H
