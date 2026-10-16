#include "sidecard_mhc_pre_v310_tiling.h"
#include <algorithm>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
  for (size_t index = 0; index < 5; ++index) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(index));
  }
  const auto residual = context->GetInputShape(0)->GetStorageShape();
  OP_CHECK_IF(residual.GetDimNum() != 3 || residual.GetDim(1) != 4 || residual.GetDim(2) != 4096,
              OP_LOGE(context, "residual must be [rows,4,4096]"), return ge::GRAPH_FAILED);
  const int64_t rows = residual.GetDim(0);
  const auto fn = context->GetInputShape(1)->GetStorageShape();
  OP_CHECK_IF(fn.GetDimNum() != 2 || fn.GetDim(0) != 24 || fn.GetDim(1) != 16384,
              OP_LOGE(context, "fn must be [24,16384]"), return ge::GRAPH_FAILED);
  const int64_t sizes[] = {3, 24, 4096};
  for (size_t index = 2; index < 5; ++index) {
    const auto input = context->GetInputShape(index)->GetStorageShape();
    OP_CHECK_IF(input.GetDimNum() != 1 || input.GetDim(0) != sizes[index - 2],
                OP_LOGE(context, "invalid coefficient or norm shape"), return ge::GRAPH_FAILED);
  }
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetAttrs());
  const auto normEps = context->GetAttrs()->GetAttrPointer<float>(0);
  const auto hcEps = context->GetAttrs()->GetAttrPointer<float>(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, normEps);
  OP_CHECK_NULL_WITH_CONTEXT(context, hcEps);
  OP_CHECK_IF(!std::isfinite(*normEps) || *normEps <= 0 || !std::isfinite(*hcEps) || *hcEps <= 0,
              OP_LOGE(context, "epsilons must be finite and positive"), return ge::GRAPH_FAILED);
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  constexpr uint64_t REQUIRED_UB_BYTES = 118816;
  uint64_t ubBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
  OP_CHECK_IF(ubBytes < REQUIRED_UB_BYTES, OP_LOGE(context, "insufficient UB for mHC pre row"),
              return ge::GRAPH_FAILED);
  // Older 310P descriptions expose AI cores with a vector pipeline rather
  // than an independent AIV count. Preserve that supported metadata path.
  const uint32_t vectorCores = platform.GetCoreNumAiv();
  const uint32_t cores = vectorCores != 0 ? vectorCores : platform.GetCoreNumAic();
  OP_CHECK_IF(cores == 0, OP_LOGE(context, "no 310P vector-capable cores"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(rows < 1 || rows > 32, OP_LOGE(context, "supported row range is 1 through 32"), return ge::GRAPH_FAILED);
  const uint32_t blocks = std::min<int64_t>(rows, cores);
  SidecardMhcPreV310TilingData data;
  data.set_rows(rows);
  data.set_blocks(blocks);
  data.set_normEps(*normEps);
  data.set_hcEps(*hcEps);
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
IMPL_OP_OPTILING(SidecardMhcPreV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
}  // namespace optiling
