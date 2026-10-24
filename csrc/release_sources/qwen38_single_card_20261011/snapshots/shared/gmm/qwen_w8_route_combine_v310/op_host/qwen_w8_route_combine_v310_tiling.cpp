// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>

#include "qwen_w8_route_combine_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr int64_t MAX_TOKENS = 8;
constexpr int64_t MAX_TOP_K = 16;
constexpr int64_t MAX_HIDDEN_SIZE = 8192;
constexpr int64_t HIDDEN_ALIGNMENT = 16;
constexpr int64_t HIDDEN_TILE = 512;

static ge::graphStatus TileRouteCombine(gert::TilingContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
  for (size_t input = 0; input < 4; ++input) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(input));
  }
  platform_ascendc::PlatformAscendC device(context->GetPlatformInfo());
  const uint32_t coreCount = device.GetCoreNumAiv();
  OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

  const auto sortedRoutes = context->GetInputShape(0)->GetStorageShape();
  const auto inverseOrder = context->GetInputShape(1)->GetStorageShape();
  const auto routeWeights = context->GetInputShape(2)->GetStorageShape();
  const auto localRouteMask = context->GetInputShape(3)->GetStorageShape();
  OP_CHECK_IF(sortedRoutes.GetDimNum() != 2 || inverseOrder.GetDimNum() != 1 ||
                  routeWeights.GetDimNum() != 2 || localRouteMask.GetDimNum() != 2,
              OP_LOGE(context, "expected [R,H], [R], [T,K], [T,K]"), return ge::GRAPH_FAILED);
  const int64_t numTokens = routeWeights.GetDim(0);
  const int64_t topK = routeWeights.GetDim(1);
  const int64_t routeCount = numTokens * topK;
  const int64_t hiddenSize = sortedRoutes.GetDim(1);
  OP_CHECK_IF(numTokens <= 0 || numTokens > MAX_TOKENS || topK <= 0 || topK > MAX_TOP_K ||
                  hiddenSize <= 0 || hiddenSize > MAX_HIDDEN_SIZE || hiddenSize % HIDDEN_ALIGNMENT != 0,
              OP_LOGE(context, "unsupported W8 route-combine dimensions"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(sortedRoutes.GetDim(0) != routeCount || inverseOrder.GetDim(0) != routeCount ||
                  localRouteMask.GetDim(0) != numTokens || localRouteMask.GetDim(1) != topK,
              OP_LOGE(context, "route-combine inputs disagree on route count"), return ge::GRAPH_FAILED);

  const int64_t tilesPerToken = (hiddenSize + HIDDEN_TILE - 1) / HIDDEN_TILE;
  const int64_t taskCount = numTokens * tilesPerToken;
  context->SetBlockDim(static_cast<uint32_t>(std::min<int64_t>(taskCount, coreCount)));
  context->SetTilingKey(0);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  workspace[0] = 0;

  QwenW8RouteCombineTilingData data;
  data.set_numTokens(numTokens);
  data.set_topK(topK);
  data.set_hiddenSize(hiddenSize);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

struct RouteCombineCompileInfo {};
static ge::graphStatus ParseRouteCombine(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
}  // namespace

IMPL_OP_OPTILING(QwenW8RouteCombineV310)
    .Tiling(TileRouteCombine)
    .TilingParse<RouteCombineCompileInfo>(ParseRouteCombine);
}  // namespace optiling
