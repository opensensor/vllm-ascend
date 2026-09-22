#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

constexpr int64_t QSA_INDEX_CACHE_SCRATCH_ROWS = 3;

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *query = context->GetInputShape(0);
    const gert::Shape *blockTable = context->GetInputShape(2);
    const gert::Shape *cache = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTable);
    OP_CHECK_NULL_WITH_CONTEXT(context, cache);
    gert::Shape *scores = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scores);
    scores->SetDimNum(2);
    scores->SetDim(0, query->GetDim(0));
    scores->SetDim(1, blockTable->GetDim(1) *
                           (cache->GetDim(1) - QSA_INDEX_CACHE_SCRATCH_ROWS));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaIndexerScoreV310).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
