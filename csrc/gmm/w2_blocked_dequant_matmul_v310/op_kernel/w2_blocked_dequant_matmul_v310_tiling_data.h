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
 * \file w2_blocked_dequant_matmul_v310_tiling_data.h
 * \brief plain tiling data struct mirror
 */

#ifndef W2_BLOCKED_DEQUANT_MATMUL_V310_TILING_DATA_H_
#define W2_BLOCKED_DEQUANT_MATMUL_V310_TILING_DATA_H_

#include <cstdint>

struct W2BlockedDequantMatmulTilingData {
    int64_t numTokens;  // T
    int64_t nDim;       // N
    int64_t kDim;       // K
};
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_TILING_DATA_H_
