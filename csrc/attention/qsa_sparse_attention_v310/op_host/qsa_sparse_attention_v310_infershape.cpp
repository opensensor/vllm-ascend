#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *query = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    gert::Shape *output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    *output = *query;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaSparseAttentionV310).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
