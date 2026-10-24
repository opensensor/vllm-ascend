// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
#include "w2_grouped_blocked_dequant_matmul_v310_tiling.h"

namespace optiling {
constexpr int64_t MAX_ROUTES = 5120;
constexpr int64_t OUTPUT_TILE = 128;
constexpr int64_t INPUT_TILE = 128;
constexpr int64_t BLOCK_SCALE = 32;
constexpr int64_t MIN_INPUT_DIM = 256;

static ge::graphStatus TileW2Grouped(gert::TilingContext* context) {
  auto platform = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platform);
  platform_ascendc::PlatformAscendC device(platform);
  for (uint32_t i = 0; i < 4; ++i) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
  }
  const auto x = context->GetInputShape(0)->GetStorageShape();
  const auto codes = context->GetInputShape(1)->GetStorageShape();
  const auto scales = context->GetInputShape(2)->GetStorageShape();
  const auto ends = context->GetInputShape(3)->GetStorageShape();
  const auto codesDesc = context->GetInputDesc(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, codesDesc);
  const bool nzPacked = codesDesc->GetDataType() == ge::DT_INT8;
  OP_CHECK_IF(codesDesc->GetDataType() != ge::DT_UINT8 && !nzPacked,
              OP_LOGE(context, "codes must be uint8 row-packed or int8 NZ-packed"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(x.GetDimNum() != 2 || codes.GetDimNum() != 3 || scales.GetDimNum() != 3 || ends.GetDimNum() != 1,
              OP_LOGE(context, "expected x[R,K], codes[E,N,packedK], scales[E,N/32,K/32], group_ends[E]"),
              return ge::GRAPH_FAILED);
  const int64_t rows = x.GetDim(0), k = x.GetDim(1), experts = codes.GetDim(0), n = codes.GetDim(1);
  const int64_t packedK = codes.GetDim(2);
  OP_CHECK_IF(packedK <= 0 || k % packedK != 0,
              OP_LOGE(context, "packed K must divide activation K"), return ge::GRAPH_FAILED);
  const int64_t codesPerByte = k / packedK;
  OP_CHECK_IF(rows <= 0 || rows > MAX_ROUTES || experts <= 0 || n <= 0 || n % OUTPUT_TILE != 0 ||
                  k < MIN_INPUT_DIM || k % INPUT_TILE != 0 || (codesPerByte != 2 && codesPerByte != 4) ||
                  ends.GetDim(0) != experts || scales.GetDim(0) != experts ||
                  scales.GetDim(1) != n / BLOCK_SCALE || scales.GetDim(2) != k / BLOCK_SCALE,
              OP_LOGE(context, "invalid grouped packed W2/W4 dimensions"), return ge::GRAPH_FAILED);
  const uint32_t cores = device.GetCoreNumAic();
  OP_CHECK_IF(cores == 0, OP_LOGE(context, "no AI cores"), return ge::GRAPH_FAILED);
  const uint32_t nBlocks = static_cast<uint32_t>(n / OUTPUT_TILE);
  // Keep every physical Cube core busy and let it advance through multiple N
  // tiles using its private packed-NZ workspace.  The W2 kernel places its
  // reusable dequant tables above CATLASS's UB epilogue region, making this
  // workspace-reuse schedule safe for routes larger than 32 rows.
  const uint32_t blocks = std::min(cores, nBlocks);
  W2GroupedBlockedDequantMatmulTilingData data;
  data.set_numRows(rows);
  data.set_numExperts(experts);
  data.set_nDim(n);
  data.set_kDim(k);
  data.set_codesPerByte(codesPerByte);
  data.set_nzPacked(nzPacked ? 1 : 0);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  workspace[0] = device.GetLibApiWorkSpaceSize() +
                 static_cast<size_t>(blocks) * OUTPUT_TILE * k * sizeof(uint16_t);
  context->SetBlockDim(blocks);
  context->SetTilingKey(0);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus ParseW2Grouped(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
struct W2GroupedCompileInfo {};
IMPL_OP_OPTILING(W2GroupedBlockedDequantMatmulV310)
    .Tiling(TileW2Grouped)
    .TilingParse<W2GroupedCompileInfo>(ParseW2Grouped);
}  // namespace optiling
