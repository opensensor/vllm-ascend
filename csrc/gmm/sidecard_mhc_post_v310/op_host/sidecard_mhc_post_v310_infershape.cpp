#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  for (size_t index = 0; index < 1; ++index) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(index));
  }
  const auto* residual = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, residual);
  *context->GetOutputShape(0) = *residual;
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SidecardMhcPostV310).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
