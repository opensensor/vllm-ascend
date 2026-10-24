// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include "qwen_w4_a8_pack_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
constexpr int64_t MAX_ROUTES = 5120, MIN_K = 256, MAX_K = 2560, GROUP_SIZE = 128, GROUPS_PER_BATCH = 8;
static ge::graphStatus TilePack(gert::TilingContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
  platform_ascendc::PlatformAscendC device(context->GetPlatformInfo());
  const auto shape = context->GetInputShape(0)->GetStorageShape();
  OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "expected FP16 [R,K]"), return ge::GRAPH_FAILED);
  const int64_t rows = shape.GetDim(0), k = shape.GetDim(1);
  OP_CHECK_IF(
      rows <= 0 || rows > MAX_ROUTES || k < MIN_K || k > MAX_K || k % GROUP_SIZE != 0 || device.GetCoreNumAic() == 0,
      OP_LOGE(context, "unsupported W4A8 pack dimensions"), return ge::GRAPH_FAILED);
  QwenW4A8PackTilingData data;
  data.set_groups(rows * (k / GROUP_SIZE));
  context->SetBlockDim(
      std::min<int64_t>(device.GetCoreNumAic(), (rows * (k / GROUP_SIZE) + GROUPS_PER_BATCH - 1) / GROUPS_PER_BATCH));
  context->SetTilingKey(0);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  workspace[0] = device.GetLibApiWorkSpaceSize();
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
struct PackCompileInfo {};
static ge::graphStatus ParsePack(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(QwenW4A8PackV310).Tiling(TilePack).TilingParse<PackCompileInfo>(ParsePack);
}  // namespace optiling
