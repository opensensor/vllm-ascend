// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *groups = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, groups);
    const int64_t *numKvHeads = context->GetAttrs()->GetInt(0);
    const int64_t *headDim = context->GetAttrs()->GetInt(1);
    const bool *transposeOutput = context->GetAttrs()->GetBool(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, numKvHeads);
    OP_CHECK_NULL_WITH_CONTEXT(context, headDim);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeOutput);
    gert::Shape *output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    const int64_t selectedTokens = groups->GetDim(1) * 4 + 4;
    const int64_t paddedTokens = (selectedTokens + 15) / 16 * 16;
    *output = *transposeOutput
                  ? gert::Shape({groups->GetDim(0), *numKvHeads, *headDim, paddedTokens})
                  : gert::Shape({groups->GetDim(0), *numKvHeads, paddedTokens, *headDim});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaGatherValueNzV310).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
