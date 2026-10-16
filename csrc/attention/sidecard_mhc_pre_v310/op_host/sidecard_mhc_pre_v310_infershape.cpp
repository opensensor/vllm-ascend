#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  for (size_t index = 0; index < 3; ++index) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(index));
  }
  const auto* input = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, input);
  OP_CHECK_IF(input->GetDimNum() != 3, OP_LOGE(context, "expected residual rank 3"), return ge::GRAPH_FAILED);
  const int64_t rows = input->GetDim(0);
  *context->GetOutputShape(0) = gert::Shape({rows, 4, 1});
  *context->GetOutputShape(1) = gert::Shape({rows, 4, 4});
  *context->GetOutputShape(2) = gert::Shape({rows, 4096});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  context->SetOutputDataType(2, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SidecardMhcPreV310).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
