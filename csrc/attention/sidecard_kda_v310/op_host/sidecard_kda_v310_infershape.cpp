#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    for (int output = 0; output < 2; ++output) {
        const auto* input = context->GetInputShape(output == 0 ? 2 : 5);
        auto* destination = context->GetOutputShape(output);
        OP_CHECK_NULL_WITH_CONTEXT(context, input);
        OP_CHECK_NULL_WITH_CONTEXT(context, destination);
        *destination = *input;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SidecardKdaV310).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
