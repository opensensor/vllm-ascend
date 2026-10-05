// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeQsaPositionMetadataAicpuV310(gert::InferShapeContext* context) {
    const auto* positions = context->GetInputShape(0);
    const auto* metadata = context->GetInputShape(1);
    if (positions == nullptr || metadata == nullptr || positions->GetDimNum() != 1 ||
        metadata->GetDimNum() != 2 || metadata->GetDim(0) != 4 ||
        positions->GetDim(0) != metadata->GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaPositionMetadataAicpuV310)
    .InferShape(InferShapeQsaPositionMetadataAicpuV310);
}  // namespace ops
