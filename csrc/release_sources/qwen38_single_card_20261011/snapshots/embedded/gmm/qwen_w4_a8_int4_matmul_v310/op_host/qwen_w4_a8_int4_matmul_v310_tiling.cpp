// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include "qwen_w4_a8_int4_matmul_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
// Grouped prefill streams route rows through bounded M tiles. The 128-entry
// route cache is used only by the separately capped int32 routed-decode path.
constexpr int64_t MAX_ROUTES = 20480;
constexpr int64_t GROUP_SIZE = 128;
constexpr int64_t OUTPUT_TILE = 16;
constexpr int64_t MIN_K = 256;
constexpr int64_t MAX_K = 2560;
constexpr int64_t MAX_N = 2 * MAX_K;
constexpr int64_t DECODE_ROUTE_LIMIT = 128;

static ge::graphStatus TileQwenW4A8Int4(gert::TilingContext* context) {
  auto platform = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platform);
  platform_ascendc::PlatformAscendC device(platform);
  for (uint32_t i = 0; i < 9; ++i) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
  }
  const auto x = context->GetInputShape(0)->GetStorageShape();
  const auto codes = context->GetInputShape(4)->GetStorageShape();
  const auto scales = context->GetInputShape(5)->GetStorageShape();
  const auto offsets = context->GetInputShape(6)->GetStorageShape();
  const auto ends = context->GetInputShape(8)->GetStorageShape();
  OP_CHECK_IF(x.GetDimNum() != 2 || codes.GetDimNum() != 3 || scales.GetDimNum() != 3 || offsets.GetDimNum() != 3 ||
                  ends.GetDimNum() != 1,
              OP_LOGE(context, "expected x[R,K], banks[E,N,*], group_ends[E]"), return ge::GRAPH_FAILED);
  const bool routed = context->GetInputDesc(8)->GetDataType() == ge::DT_INT32;
  const int64_t input_rows = x.GetDim(0), rows = routed ? ends.GetDim(0) : input_rows;
  const int64_t k = x.GetDim(1) * 2, experts = codes.GetDim(0), n = codes.GetDim(1);
  OP_CHECK_IF(rows <= 0 || rows > MAX_ROUTES || experts <= 0 || n <= 0 || n > MAX_N || n % GROUP_SIZE != 0 ||
                  k < MIN_K || k > MAX_K || k % GROUP_SIZE != 0 || codes.GetDim(2) * 2 != k || input_rows <= 0 ||
                  rows < input_rows || rows % input_rows != 0 || (!routed && ends.GetDim(0) != experts),
              OP_LOGE(context, "invalid Qwen W4 grouped dimensions"), return ge::GRAPH_FAILED);
  for (const auto& shape : {scales, offsets}) {
    OP_CHECK_IF(shape.GetDim(0) != experts || shape.GetDim(1) != n || shape.GetDim(2) != k / GROUP_SIZE,
                OP_LOGE(context, "W4 metadata must be [E,N,K/128]"), return ge::GRAPH_FAILED);
  }
  const auto high = context->GetInputShape(1)->GetStorageShape();
  const auto xs = context->GetInputShape(2)->GetStorageShape();
  const auto sums = context->GetInputShape(3)->GetStorageShape();
  const auto ws = context->GetInputShape(7)->GetStorageShape();
  OP_CHECK_IF(high.GetDimNum() != 2 || (xs.GetDimNum() != 2 && xs.GetDimNum() != 3) ||
                  sums.GetDimNum() != xs.GetDimNum() || ws.GetDimNum() != 3,
              OP_LOGE(context, "invalid W4A8 auxiliary ranks"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(high.GetDim(0) != input_rows || high.GetDim(1) != k / 2 || xs.GetDim(0) != input_rows ||
                  xs.GetDim(1) != k / GROUP_SIZE || sums.GetDim(0) != input_rows || sums.GetDim(1) != k / GROUP_SIZE ||
                  ws.GetDim(0) != experts || ws.GetDim(1) != n || ws.GetDim(2) != k / GROUP_SIZE,
              OP_LOGE(context, "invalid W4A8 auxiliary dimensions"), return ge::GRAPH_FAILED);
  const int64_t lanes = xs.GetDimNum() == 3 ? 8 : 1;
  OP_CHECK_IF(lanes == 8 && (xs.GetDim(2) != 8 || sums.GetDim(2) != 8),
              OP_LOGE(context, "W4A8 broadcast metadata requires eight lanes"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(routed && (rows > DECODE_ROUTE_LIMIT || lanes != 8),
              OP_LOGE(context, "native routed decode requires <=128 rows and broadcast metadata"),
              return ge::GRAPH_FAILED);
  const uint32_t cores = device.GetCoreNumAic();
  OP_CHECK_IF(cores == 0, OP_LOGE(context, "no AI cores"), return ge::GRAPH_FAILED);
  const uint32_t blocks = std::min<int64_t>(cores, experts * (n / OUTPUT_TILE));
  QwenW4A8Int4MatmulTilingData data;
  data.set_numRows(rows);
  data.set_numExperts(experts);
  data.set_nDim(n);
  data.set_kDim(k);
  data.set_metadataLanes(lanes);
  data.set_routed(routed);
  data.set_broadcastFactor(rows / input_rows);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  // All native INT4 tiles and accumulators are on chip.
  workspace[0] = device.GetLibApiWorkSpaceSize();
  context->SetBlockDim(blocks);
  context->SetTilingKey(0);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus ParseQwenW4A8Int4(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
struct QwenW4A8Int4CompileInfo {};
IMPL_OP_OPTILING(QwenW4A8Int4MatmulV310)
    .Tiling(TileQwenW4A8Int4)
    .TilingParse<QwenW4A8Int4CompileInfo>(ParseQwenW4A8Int4);
}  // namespace optiling
