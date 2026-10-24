// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
static ge::graphStatus InferRouteCombine(gert::InferShapeContext* context) {
  auto sortedRoutes = context->GetInputShape(0);
  auto routeWeights = context->GetInputShape(2);
  auto output = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, sortedRoutes);
  OP_CHECK_NULL_WITH_CONTEXT(context, routeWeights);
  OP_CHECK_NULL_WITH_CONTEXT(context, output);
  OP_CHECK_IF(sortedRoutes->GetDimNum() != 2 || routeWeights->GetDimNum() != 2,
              OP_LOGE(context, "expected sorted routes [R,H] and route weights [T,K]"),
              return ge::GRAPH_FAILED);
  output->SetDimNum(2);
  output->SetDim(0, routeWeights->GetDim(0));
  output->SetDim(1, sortedRoutes->GetDim(1));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DtypeRouteCombine(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QwenW8RouteCombineV310)
    .InferShape(InferRouteCombine)
    .InferDataType(DtypeRouteCombine);
}  // namespace ops
