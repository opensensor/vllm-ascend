// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "qsa_gather_value_nz_v310_tiling.h"

#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t NZ_INNER = 16;
constexpr int64_t COMPRESS_RATIO = 4;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

    const auto cache = context->GetInputShape(0)->GetStorageShape();
    const auto groups = context->GetInputShape(1)->GetStorageShape();
    const auto counts = context->GetInputShape(2)->GetStorageShape();
    const auto table = context->GetInputShape(5)->GetStorageShape();
    const auto output = context->GetOutputShape(0)->GetStorageShape();
    const bool *transposeOutput = context->GetAttrs()->GetBool(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeOutput);
    OP_CHECK_IF(cache.GetDimNum() != 4 || groups.GetDimNum() != 2 || counts.GetDimNum() != 1 ||
                    table.GetDimNum() != 2 || output.GetDimNum() != 4,
                OP_LOGE(context, "invalid QSA gather tensor rank"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDim(3) != NZ_INNER || cache.GetDim(2) <= 0 ||
                    cache.GetDim(2) % COMPRESS_RATIO != 0 || table.GetDim(0) != 1,
                OP_LOGE(context, "QSA gather requires a 16-wide NZ cache and one block table"),
                return ge::GRAPH_FAILED);
    const int64_t numTokens = groups.GetDim(0);
    const int64_t numKvHeads = output.GetDim(1);
    const int64_t headDim = output.GetDim(*transposeOutput ? 2 : 3);
    const int64_t outputTokens = output.GetDim(*transposeOutput ? 3 : 2);
    OP_CHECK_IF(numTokens <= 0 || numKvHeads <= 0 || headDim <= 0 || headDim % NZ_INNER != 0 ||
                    counts.GetDim(0) != numTokens || cache.GetDim(1) != numKvHeads * headDim / NZ_INNER,
                OP_LOGE(context, "QSA gather cache/selection/output dimensions mismatch"), return ge::GRAPH_FAILED);
    const int64_t expectedTokens = ((groups.GetDim(1) * COMPRESS_RATIO + COMPRESS_RATIO + NZ_INNER - 1)
                                    / NZ_INNER) * NZ_INNER;
    OP_CHECK_IF(output.GetDim(0) != numTokens || outputTokens != expectedTokens,
                OP_LOGE(context, "QSA gather output must have padded selected-token width"),
                return ge::GRAPH_FAILED);

    // Keep all head-dimension blocks of one token/head on the same core. The
    // group list and counts are then loaded once and reused across D blocks.
    const int64_t taskCount = numTokens * numKvHeads;
    const uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(taskCount, coreCount));
    QsaGatherValueNzV310TilingData data;
    data.set_numTokens(numTokens);
    data.set_numKvHeads(numKvHeads);
    data.set_headDimBlocks(headDim / NZ_INNER);
    data.set_cacheHeadDimBlocks(cache.GetDim(1));
    data.set_cacheBlockSize(cache.GetDim(2));
    data.set_selectedGroupsWidth(groups.GetDim(1));
    data.set_outputTokenBlocks(expectedTokens / NZ_INNER);
    data.set_maxBlocksPerSequence(table.GetDim(1));
    data.set_taskCount(taskCount);
    data.set_tasksPerCore((taskCount + blockDim - 1) / blockDim);
    data.set_transposeOutput(*transposeOutput ? 1 : 0);
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Parse(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

struct CompileInfo {};

}  // namespace

IMPL_OP_OPTILING(QsaGatherValueNzV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
