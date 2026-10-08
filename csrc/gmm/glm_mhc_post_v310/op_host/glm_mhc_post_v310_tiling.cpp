// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "../mhc_post_geometry.h"
#include "glm_mhc_post_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
namespace optiling {
static ge::graphStatus TileMhcPost(gert::TilingContext* context) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
    const auto shape = context->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "expected matrix"), return ge::GRAPH_FAILED);
    for (size_t index = 1; index < 4; ++index) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(index));
        const auto other = context->GetInputShape(index)->GetStorageShape();
        const int64_t last = index == 1 ? shape.GetDim(1) : (index == 2 ? 1 : NsGlmMhcPost::STREAMS);
        OP_CHECK_IF(other.GetDimNum() != 3 || other.GetDim(0) != shape.GetDim(0) ||
                    other.GetDim(1) != NsGlmMhcPost::STREAMS || other.GetDim(2) != last,
                    OP_LOGE(context, "invalid residual or mixing shape"), return ge::GRAPH_FAILED);
    }
    platform_ascendc::PlatformAscendC device(context->GetPlatformInfo());
    OP_CHECK_IF(!NsGlmMhcPost::ValidGeometry(shape.GetDim(0), shape.GetDim(1)) || device.GetCoreNumAic() == 0,
                OP_LOGE(context, "invalid mHC post geometry"), return ge::GRAPH_FAILED);
    GlmMhcPostTilingData data;
    data.set_rows(shape.GetDim(0));
    data.set_width(shape.GetDim(1));
    const int64_t tasks = shape.GetDim(0) * ((shape.GetDim(1) + NsGlmMhcPost::TILE - 1) / NsGlmMhcPost::TILE);
    context->SetBlockDim(std::min<int64_t>(device.GetCoreNumAic(), tasks));
    context->SetTilingKey(0);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = device.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
struct MhcPostCompileInfo {};
static ge::graphStatus ParseMhcPost(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(GlmMhcPostV310).Tiling(TileMhcPost).TilingParse<MhcPostCompileInfo>(ParseMhcPost);
}  // namespace optiling
