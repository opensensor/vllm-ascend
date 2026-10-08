// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "../route_combine_geometry.h"
#include "w2_route_combine_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
namespace optiling {
static ge::graphStatus TileRouteCombine(gert::TilingContext* context) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    for (uint32_t i = 0; i < 4; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
    }
    const auto routed = context->GetInputShape(0)->GetStorageShape();
    const auto inverse = context->GetInputShape(1)->GetStorageShape();
    const auto weights = context->GetInputShape(2)->GetStorageShape();
    const auto ends = context->GetInputShape(3)->GetStorageShape();
    OP_CHECK_IF(routed.GetDimNum() != 2 || inverse.GetDimNum() != 1 ||
                    weights.GetDimNum() != 2 || ends.GetDimNum() != 1,
                OP_LOGE(context, "invalid route-combine ranks"), return ge::GRAPH_FAILED);
    const int64_t rows = routed.GetDim(0), hidden = routed.GetDim(1);
    const int64_t tokens = weights.GetDim(0), topK = weights.GetDim(1), experts = ends.GetDim(0);
    platform_ascendc::PlatformAscendC device(context->GetPlatformInfo());
    OP_CHECK_IF(!NsW2Combine::ValidGeometry(rows, hidden, tokens, topK, experts) ||
                    inverse.GetDim(0) != rows || device.GetCoreNumAic() == 0,
                OP_LOGE(context, "invalid route-combine geometry"), return ge::GRAPH_FAILED);
    W2RouteCombineTilingData data;
    data.set_tokens(tokens);
    data.set_hidden(hidden);
    data.set_top_k(topK);
    data.set_experts(experts);
    const int64_t tiles = (hidden + NsW2Combine::CHANNEL_TILE - 1) / NsW2Combine::CHANNEL_TILE;
    context->SetBlockDim(std::min<int64_t>(device.GetCoreNumAic(), tokens * tiles));
    context->SetTilingKey(0);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = device.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
struct RouteCombineCompileInfo {};
static ge::graphStatus ParseRouteCombine(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(W2RouteCombineV310).Tiling(TileRouteCombine).TilingParse<RouteCombineCompileInfo>(ParseRouteCombine);
}  // namespace optiling
