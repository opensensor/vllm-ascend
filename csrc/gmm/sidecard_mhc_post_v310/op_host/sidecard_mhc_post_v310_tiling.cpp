#include "sidecard_mhc_post_v310_tiling.h"
#include <algorithm>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
  for (size_t index = 0; index < 4; ++index) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(index));
  }
  const auto x = context->GetInputShape(0)->GetStorageShape();
  OP_CHECK_IF(x.GetDimNum() != 2 || x.GetDim(1) != 4096, OP_LOGE(context, "x must be [rows,4096]"),
              return ge::GRAPH_FAILED);
  const int64_t rows = x.GetDim(0);
  const int64_t sizes[] = {4096, 1, 4};
  for (size_t index = 1; index < 4; ++index) {
    const auto input = context->GetInputShape(index)->GetStorageShape();
    OP_CHECK_IF(input.GetDimNum() != 3 || input.GetDim(0) != rows || input.GetDim(1) != 4 ||
                    input.GetDim(2) != sizes[index - 1],
                OP_LOGE(context, "invalid residual or mixer shape"), return ge::GRAPH_FAILED);
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  constexpr uint64_t REQUIRED_UB_BYTES = 9216;
  uint64_t ubBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
  OP_CHECK_IF(ubBytes < REQUIRED_UB_BYTES, OP_LOGE(context, "insufficient UB for mHC post tile"),
              return ge::GRAPH_FAILED);
  // Older 310P descriptions expose AI cores with a vector pipeline rather
  // than an independent AIV count. Preserve that supported metadata path.
  const uint32_t vectorCores = platform.GetCoreNumAiv();
  const uint32_t cores = vectorCores != 0 ? vectorCores : platform.GetCoreNumAic();
  OP_CHECK_IF(cores == 0, OP_LOGE(context, "no 310P vector-capable cores"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(rows < 1 || rows > 32, OP_LOGE(context, "supported row range is 1 through 32"), return ge::GRAPH_FAILED);
  const uint32_t blocks = std::min<int64_t>(rows * 4 * 8, cores);
  SidecardMhcPostV310TilingData data;
  data.set_rows(rows);
  data.set_blocks(blocks);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  workspace[0] = 0;
  context->SetBlockDim(blocks);
  context->SetTilingKey(0);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(SidecardMhcPostV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
}  // namespace optiling
