// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeQsaExactTopkAicpuV310(gert::InferShapeContext* context) {
    const auto* scores = context->GetInputShape(0);
    const auto* indices = context->GetInputShape(1);
    if (scores == nullptr || indices == nullptr || scores->GetDimNum() != 2 ||
        indices->GetDimNum() != 2 || scores->GetDim(0) != indices->GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaExactTopkAicpuV310)
    .InferShape(InferShapeQsaExactTopkAicpuV310);
}  // namespace ops
