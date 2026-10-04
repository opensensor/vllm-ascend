#include "mhc_bf16_round_v310_tiling.h"

#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t TILE_ELEMENTS = 1024;
constexpr uint32_t MAX_BLOCKS = 8;

ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto input = context->GetInputShape(0)->GetStorageShape();
    const int64_t numel = input.GetShapeSize();
    OP_CHECK_IF(numel <= 0,
                OP_LOGE(context, "input must have at least one element"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0,
                OP_LOGE(context, "AIV core count is zero"),
                return ge::GRAPH_FAILED);
    const uint32_t blocks = static_cast<uint32_t>(
        std::min<int64_t>({(numel + TILE_ELEMENTS - 1) / TILE_ELEMENTS,
                           static_cast<int64_t>(coreCount), MAX_BLOCKS}));

    MhcBf16RoundV310TilingData data;
    data.set_numel(numel);
    data.set_blockCount(blocks);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(blocks);
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
struct CompileInfo {};

}  // namespace

IMPL_OP_OPTILING(MhcBf16RoundV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
