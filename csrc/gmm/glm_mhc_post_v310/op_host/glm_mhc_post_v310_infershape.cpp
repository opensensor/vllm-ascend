// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferMhcPost(gert::InferShapeContext* context) {
    auto input = context->GetInputShape(1);
    auto output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_CHECK_IF(input->GetDimNum() != 3, OP_LOGE(context, "expected matrix"), return ge::GRAPH_FAILED);
    *output = *input;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeMhcPost(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GlmMhcPostV310).InferShape(InferMhcPost).InferDataType(DtypeMhcPost);
}  // namespace ops
