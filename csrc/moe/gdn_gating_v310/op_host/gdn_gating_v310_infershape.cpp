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
 * \file gdn_gating_v310_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t TILE_ROWS = 32;

static ge::graphStatus InferShapeGdnGating(gert::InferShapeContext *context)
{
    const gert::Shape *aShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);

    gert::Shape *gShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gShape);
    gert::Shape *betaShape = context->GetOutputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);

    // UB->GM stores always move a complete aligned tile. ACLNN allocates this
    // padded shape; the torch adapter narrows it back to the input row count.
    *gShape = *aShape;
    *betaShape = *aShape;
    const int64_t numRows = aShape->GetDim(IDX_0);
    const int64_t paddedRows = (numRows + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS;
    gShape->SetDim(IDX_0, paddedRows);
    betaShape->SetDim(IDX_0, paddedRows);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGdnGating(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(IDX_0, ge::DT_FLOAT);
    context->SetOutputDataType(IDX_1, ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnGatingV310).InferShape(InferShapeGdnGating).InferDataType(InferDataTypeGdnGating);
}  // namespace ops
