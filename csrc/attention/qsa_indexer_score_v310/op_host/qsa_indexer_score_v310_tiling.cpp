#include "qsa_indexer_score_v310_tiling.h"

#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t QSA_COMPRESS_RATIO = 4;
constexpr int64_t QSA_INDEX_CACHE_SCRATCH_ROWS = QSA_COMPRESS_RATIO - 1;
constexpr int64_t MAX_INDEX_HEAD_DIM = 256;
constexpr int64_t SCORE_CACHE_LINE_ELEMENTS = 64 / sizeof(float);

ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

    const auto query = context->GetInputShape(0)->GetStorageShape();
    const auto cache = context->GetInputShape(1)->GetStorageShape();
    const auto blockTable = context->GetInputShape(2)->GetStorageShape();
    const auto queryStartLoc = context->GetInputShape(3)->GetStorageShape();
    OP_CHECK_IF(query.GetDimNum() != 3, OP_LOGE(context, "query must be [T,H,D]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDimNum() != 3, OP_LOGE(context, "compressed cache must be [blocks,rows,D]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockTable.GetDimNum() != 2, OP_LOGE(context, "block table must be 2D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryStartLoc.GetDimNum() != 1, OP_LOGE(context, "query start locations must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(query.GetDim(2) != cache.GetDim(2), OP_LOGE(context, "query/cache head dimensions differ"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDim(1) <= QSA_INDEX_CACHE_SCRATCH_ROWS,
                OP_LOGE(context, "compressed cache must contain group rows and three scratch rows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(query.GetDim(2) <= 0 || query.GetDim(2) > MAX_INDEX_HEAD_DIM || query.GetDim(2) % 16 != 0,
                OP_LOGE(context, "index head dimension must be a multiple of 16 up to 256"),
                return ge::GRAPH_FAILED);
    const int64_t *compressRatio = context->GetAttrs()->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, compressRatio);
    OP_CHECK_IF(*compressRatio != QSA_COMPRESS_RATIO,
                OP_LOGE(context, "310P QSA score kernel requires compression ratio 4"),
                return ge::GRAPH_FAILED);

    const int64_t groupsPerBlock = cache.GetDim(1) - QSA_INDEX_CACHE_SCRATCH_ROWS;
    const int64_t maxGroups = blockTable.GetDim(1) * groupsPerBlock;
    const int64_t taskCount = query.GetDim(0) * maxGroups;
    // The kernel writes scores with GlobalTensor::SetValue. A cache line must
    // belong to one core; otherwise the final partial line of one core can
    // race with the first line of the next core and lose scores.
    const uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(
        1, std::min<int64_t>((taskCount + SCORE_CACHE_LINE_ELEMENTS - 1) / SCORE_CACHE_LINE_ELEMENTS, coreCount)));
    const int64_t tasksPerCore = ((taskCount + blockDim - 1) / blockDim + SCORE_CACHE_LINE_ELEMENTS - 1) /
                                 SCORE_CACHE_LINE_ELEMENTS * SCORE_CACHE_LINE_ELEMENTS;
    QsaIndexerScoreV310TilingData data;
    data.set_numTokens(query.GetDim(0));
    data.set_numHeads(query.GetDim(1));
    data.set_headDim(query.GetDim(2));
    data.set_cacheRowsPerBlock(cache.GetDim(1));
    data.set_groupsPerBlock(groupsPerBlock);
    data.set_maxBlocksPerSequence(blockTable.GetDim(1));
    data.set_maxGroupsPerSequence(maxGroups);
    data.set_numRequests(queryStartLoc.GetDim(0) - 1);
    data.set_taskCount(taskCount);
    data.set_tasksPerCore(tasksPerCore);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
struct CompileInfo {};

}  // namespace

IMPL_OP_OPTILING(QsaIndexerScoreV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
