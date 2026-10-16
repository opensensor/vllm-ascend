#include "sidecard_kda_v310_tiling.h"
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t VALUE_GROUP = 16;
ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* query = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    const auto& q = query->GetStorageShape();
    OP_CHECK_IF(q.GetDimNum() != 3 || q.GetDim(0) != 1 || q.GetDim(2) != HEAD_DIM,
                OP_LOGE(context, "expected query [1,H,128]"), return ge::GRAPH_FAILED);
    const int64_t heads = q.GetDim(1);
    OP_CHECK_IF(heads < 8 || heads > 128 || heads % 8 != 0,
                OP_LOGE(context, "expected 8..128 heads divisible by 8"), return ge::GRAPH_FAILED);
    for (int index = 1; index <= 5; ++index) {
        const auto* input = context->GetInputShape(index);
        OP_CHECK_NULL_WITH_CONTEXT(context, input);
        const auto& shape = input->GetStorageShape();
        bool valid = false;
        if (index <= 3) {
            valid = shape.GetDimNum() == 3 && shape.GetDim(0) == 1 &&
                    shape.GetDim(1) == heads && shape.GetDim(2) == HEAD_DIM;
        } else if (index == 4) {
            valid = shape.GetDimNum() == 2 && shape.GetDim(0) == 1 && shape.GetDim(1) == heads;
        } else {
            valid = shape.GetDimNum() == 3 && shape.GetDim(0) == heads &&
                    shape.GetDim(1) == HEAD_DIM && shape.GetDim(2) == HEAD_DIM;
        }
        OP_CHECK_IF(!valid, OP_LOGE(context, "KDA input geometry mismatch"), return ge::GRAPH_FAILED);
    }
    auto* info = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, info);
    auto platform = platform_ascendc::PlatformAscendC(info);
    // The existing v310 recurrence uses GetCoreNumAiv. Older 310P platform
    // descriptions can report only AI cores (with a vector pipeline).
    const uint32_t vectorCores = platform.GetCoreNumAiv();
    const uint32_t cores = vectorCores != 0 ? vectorCores : platform.GetCoreNumAic();
    OP_CHECK_IF(cores == 0, OP_LOGE(context, "no 310P vector-capable cores"), return ge::GRAPH_FAILED);
    const uint32_t blocks = std::min<uint32_t>(cores, heads * (HEAD_DIM / VALUE_GROUP));
    SidecardKdaV310TilingData data;
    data.set_heads(heads);
    data.set_blocks(blocks);
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(blocks);
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
struct CompileInfo {};
}  // namespace
IMPL_OP_OPTILING(SidecardKdaV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
}  // namespace optiling
