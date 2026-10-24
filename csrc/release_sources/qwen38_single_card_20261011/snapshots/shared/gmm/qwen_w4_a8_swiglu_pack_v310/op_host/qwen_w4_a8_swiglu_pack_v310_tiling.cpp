// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include "qwen_w4_a8_swiglu_pack_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
constexpr int64_t MAX_ROUTES = 128, MIN_K = 256, MAX_K = 2560;
constexpr int64_t GROUP_SIZE = 128, GROUPS_PER_BATCH = 8;

static ge::graphStatus TileSwigluPack(gert::TilingContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
  platform_ascendc::PlatformAscendC device(context->GetPlatformInfo());
  const auto shape = context->GetInputShape(0)->GetStorageShape();
  OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "expected FP16 [R,2K]"), return ge::GRAPH_FAILED);
  const int64_t rows = shape.GetDim(0), gate_up_width = shape.GetDim(1);
  const int64_t k = gate_up_width / 2;
  OP_CHECK_IF(rows <= 0 || rows > MAX_ROUTES || gate_up_width % 2 || k < MIN_K || k > MAX_K ||
                  k % GROUP_SIZE != 0 || device.GetCoreNumAic() == 0,
              OP_LOGE(context, "unsupported W4A8 SwiGLU pack dimensions"), return ge::GRAPH_FAILED);
  const int64_t groups_per_row = k / GROUP_SIZE;
  const int64_t batches_per_row = (groups_per_row + GROUPS_PER_BATCH - 1) / GROUPS_PER_BATCH;
  QwenW4A8SwigluPackTilingData data;
  data.set_rows(rows);
  data.set_groups_per_row(groups_per_row);
  context->SetBlockDim(std::min<int64_t>(device.GetCoreNumAic(), rows * batches_per_row));
  context->SetTilingKey(0);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  workspace[0] = device.GetLibApiWorkSpaceSize();
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

struct SwigluPackCompileInfo {};
static ge::graphStatus ParseSwigluPack(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(QwenW4A8SwigluPackV310)
    .Tiling(TileSwigluPack)
    .TilingParse<SwigluPackCompileInfo>(ParseSwigluPack);
}  // namespace optiling
