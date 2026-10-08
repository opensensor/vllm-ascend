// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferRouteCombine(gert::InferShapeContext* context) {
    auto routed = context->GetInputShape(0);
    auto weights = context->GetInputShape(2);
    auto output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, routed);
    OP_CHECK_NULL_WITH_CONTEXT(context, weights);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_CHECK_IF(routed->GetDimNum() != 2 || weights->GetDimNum() != 2,
                OP_LOGE(context, "expected matrices"), return ge::GRAPH_FAILED);
    output->SetDimNum(2);
    output->SetDim(0, weights->GetDim(0));
    output->SetDim(1, routed->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeRouteCombine(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(W2RouteCombineV310).InferShape(InferRouteCombine).InferDataType(DtypeRouteCombine);
}  // namespace ops
