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
 * \file w2_blocked_dequant_matmul_v310_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_X = 0;
static constexpr int64_t IDX_CODES = 1;

// y[T, N] = x[T, K] @ (codes[N, K] * scale)^T
static ge::graphStatus InferShapeW2BlockedDequantMatmul(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *codesShape = context->GetInputShape(IDX_CODES);
    OP_CHECK_NULL_WITH_CONTEXT(context, codesShape);

    gert::Shape *yShape = context->GetOutputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));      // T
    yShape->SetDim(1, codesShape->GetDim(0));  // N

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeW2BlockedDequantMatmul(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(IDX_X, context->GetInputDataType(IDX_X));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(W2BlockedDequantMatmulV310)
    .InferShape(InferShapeW2BlockedDequantMatmul)
    .InferDataType(InferDataTypeW2BlockedDequantMatmul);
}  // namespace ops
