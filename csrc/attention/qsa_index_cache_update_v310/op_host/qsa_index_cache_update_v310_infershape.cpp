#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *cache = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, cache);
    gert::Shape *cacheOut = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheOut);
    *cacheOut = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaIndexCacheUpdateV310).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
