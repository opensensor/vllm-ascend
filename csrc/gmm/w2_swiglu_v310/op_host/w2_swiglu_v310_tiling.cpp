// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "../swiglu_geometry.h"
#include "w2_swiglu_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
namespace optiling {
static ge::graphStatus TileSwiglu(gert::TilingContext* context) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
    const auto shape = context->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "expected matrix"), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC device(context->GetPlatformInfo());
    OP_CHECK_IF(!NsW2Swiglu::ValidGeometry(shape.GetDim(0), shape.GetDim(1)) || device.GetCoreNumAic() == 0,
                OP_LOGE(context, "invalid SwiGLU geometry"), return ge::GRAPH_FAILED);
    W2SwigluTilingData data;
    data.set_rows(shape.GetDim(0));
    data.set_width(shape.GetDim(1) / 2);
    const int64_t tasks = shape.GetDim(0) * ((shape.GetDim(1) / 2 + NsW2Swiglu::TILE - 1) / NsW2Swiglu::TILE);
    context->SetBlockDim(std::min<int64_t>(device.GetCoreNumAic(), tasks));
    context->SetTilingKey(0);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = device.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
struct SwigluCompileInfo {};
static ge::graphStatus ParseSwiglu(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(W2SwigluV310).Tiling(TileSwiglu).TilingParse<SwigluCompileInfo>(ParseSwiglu);
}  // namespace optiling
