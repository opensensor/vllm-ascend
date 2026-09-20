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
 * \file w2_blocked_dequant_matmul_v310.cpp
 * \brief
 */

#include "compat_310p.h"
#include "w2_blocked_dequant_matmul_v310.h"

extern "C" __global__ __aicore__ void w2_blocked_dequant_matmul_v310(GM_ADDR x, GM_ADDR codes, GM_ADDR blockScale,
                                                                    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    NsW2::W2BlockedDequantMatmulV310Cube op;
    op.Init(x, codes, blockScale, y, user, tiling);
    op.Process();
}
