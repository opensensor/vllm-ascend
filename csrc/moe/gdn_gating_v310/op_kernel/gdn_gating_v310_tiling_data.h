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
 * \file gdn_gating_v310_tiling_data.h
 * \brief plain tiling data struct mirror
 */
#ifndef GDN_GATING_V310_TILING_DATA_H_
#define GDN_GATING_V310_TILING_DATA_H_

#include <cstdint>

struct GdnGatingTilingData {
    int64_t numRows;
    int64_t numHeads;
    int64_t tileRows;
    int64_t tilesPerCore;
    int64_t tileCount;
    float beta;
    float invBeta;
};
#endif  // GDN_GATING_V310_TILING_DATA_H_
