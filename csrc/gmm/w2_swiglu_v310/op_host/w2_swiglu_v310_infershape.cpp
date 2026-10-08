// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferSwiglu(gert::InferShapeContext* context) {
    auto input = context->GetInputShape(0);
    auto output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_CHECK_IF(input->GetDimNum() != 2, OP_LOGE(context, "expected matrix"), return ge::GRAPH_FAILED);
    output->SetDimNum(2);
    output->SetDim(0, input->GetDim(0));
    output->SetDim(1, input->GetDim(1) < 0 ? -1 : input->GetDim(1) / 2);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeSwiglu(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(W2SwigluV310).InferShape(InferSwiglu).InferDataType(DtypeSwiglu);
}  // namespace ops
