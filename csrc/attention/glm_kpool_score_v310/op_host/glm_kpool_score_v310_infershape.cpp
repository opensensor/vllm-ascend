// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext* c) {
    const auto q = c->GetInputShape(0); auto out = c->GetOutputShape(0);
    const auto pools = c->GetAttrs()->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(c, q); OP_CHECK_NULL_WITH_CONTEXT(c, out); OP_CHECK_NULL_WITH_CONTEXT(c, pools);
    out->SetDimNum(2); out->SetDim(0, q->GetDim(0)); out->SetDim(1, *pools);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext* c) { c->SetOutputDataType(0, ge::DT_FLOAT); return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(GlmKpoolScoreV310).InferShape(Infer).InferDataType(Dtype);
}
