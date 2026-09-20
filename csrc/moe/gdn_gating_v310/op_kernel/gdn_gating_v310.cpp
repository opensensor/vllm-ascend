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
 * \file gdn_gating_v310.cpp
 * \brief
 */

#include "gdn_gating_v310.h"

namespace {

template <typename T>
__aicore__ inline void RunGdnGating(GM_ADDR a, GM_ADDR b, GM_ADDR negExpALogTiled, GM_ADDR dtBiasTiled, GM_ADDR g,
                                    GM_ADDR betaOut, const GdnGatingTilingData *tilingData)
{
    AscendC::TPipe pipe;
    NsGdnGating::GdnGatingV310<T> op;
    op.Init(a, b, negExpALogTiled, dtBiasTiled, g, betaOut, tilingData, &pipe);
    op.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void gdn_gating_v310(GM_ADDR a, GM_ADDR b, GM_ADDR negExpALogTiled,
                                                      GM_ADDR dtBiasTiled, GM_ADDR g, GM_ADDR betaOut,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    // This operator only uses vector instructions and UB<->GM transfers, so it
    // must be scheduled as an AIV task. Do not add an AIC early return here:
    // 310P may report its unified core through g_coreType while executing the
    // vector task selected by this annotation.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(GdnGatingTilingData);
    GET_TILING_DATA_WITH_STRUCT(GdnGatingTilingData, tilingData, tiling);

    RunGdnGating<half>(a, b, negExpALogTiled, dtBiasTiled, g, betaOut, &tilingData);
}
