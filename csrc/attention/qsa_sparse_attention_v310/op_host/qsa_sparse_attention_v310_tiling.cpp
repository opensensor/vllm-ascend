#include "qsa_sparse_attention_v310_tiling.h"

#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr uint32_t QUERY = 0;
constexpr uint32_t KEY_CACHE = 1;
constexpr uint32_t GROUP_INDICES = 3;
constexpr uint32_t BLOCK_TABLE = 7;
constexpr uint32_t QUERY_START_LOC = 8;
constexpr int64_t NZ_INNER = 16;
constexpr int64_t QSA_COMPRESS_RATIO = 4;
constexpr int64_t MAX_HEAD_DIM = 256;
constexpr int64_t MAX_QUERY_HEADS_PER_KV_HEAD = 24;
constexpr int64_t MAX_DATA_COPY_BLOCK_SIZE = 65535;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

    const auto query = context->GetInputShape(QUERY)->GetStorageShape();
    const auto cache = context->GetInputShape(KEY_CACHE)->GetStorageShape();
    const auto groups = context->GetInputShape(GROUP_INDICES)->GetStorageShape();
    const auto blockTable = context->GetInputShape(BLOCK_TABLE)->GetStorageShape();
    const auto queryStartLoc = context->GetInputShape(QUERY_START_LOC)->GetStorageShape();
    OP_CHECK_IF(query.GetDimNum() != 3, OP_LOGE(context, "query must be [T, Nq, D]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDimNum() != 4, OP_LOGE(context, "NZ cache must be [blocks, HD/16, block, 16]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groups.GetDimNum() != 2, OP_LOGE(context, "groupIndices must be [T, K]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockTable.GetDimNum() != 2, OP_LOGE(context, "blockTable must be 2D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryStartLoc.GetDimNum() != 1, OP_LOGE(context, "queryStartLoc must be 1D"),
                return ge::GRAPH_FAILED);

    const int64_t numTokens = query.GetDim(0);
    const int64_t numQueryHeads = query.GetDim(1);
    OP_CHECK_IF(numTokens <= 0 || numQueryHeads <= 0,
                OP_LOGE(context, "query must contain tokens and heads"), return ge::GRAPH_FAILED);
    const int64_t headDim = query.GetDim(2);
    const int64_t cacheHeadDimBlocks = cache.GetDim(1);
    const int64_t cacheBlockSize = cache.GetDim(2);
    OP_CHECK_IF(cacheBlockSize <= 0 || cacheBlockSize % QSA_COMPRESS_RATIO != 0 ||
                    cacheBlockSize > MAX_DATA_COPY_BLOCK_SIZE,
                OP_LOGE(context, "cache block size must be a positive multiple of compression ratio up to 65535"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headDim <= 0 || headDim > MAX_HEAD_DIM || headDim % NZ_INNER != 0,
                OP_LOGE(context, "head dimension must be a positive multiple of 16 up to 256"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDim(3) != NZ_INNER, OP_LOGE(context, "NZ cache inner dimension must be 16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cacheHeadDimBlocks % (headDim / NZ_INNER) != 0,
                OP_LOGE(context, "cache head dimension is incompatible with query head dimension"),
                return ge::GRAPH_FAILED);
    const int64_t numKvHeads = cacheHeadDimBlocks / (headDim / NZ_INNER);
    OP_CHECK_IF(numKvHeads <= 0, OP_LOGE(context, "cache must contain at least one KV head"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numQueryHeads % numKvHeads != 0 || numQueryHeads / numKvHeads > MAX_QUERY_HEADS_PER_KV_HEAD,
                OP_LOGE(context, "query heads per KV head must be between 1 and 24"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(groups.GetDim(0) != numTokens, OP_LOGE(context, "selection rows must equal query tokens"),
                return ge::GRAPH_FAILED);
    const int64_t *compressRatio = context->GetAttrs()->GetInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, compressRatio);
    OP_CHECK_IF(*compressRatio != QSA_COMPRESS_RATIO,
                OP_LOGE(context, "310P QSA kernel currently requires compression ratio 4"),
                return ge::GRAPH_FAILED);
    const int64_t *scaleQ24 = context->GetAttrs()->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleQ24);

    // Keep all query heads together for large prefills. A small decode would
    // otherwise launch only numKvHeads tasks and leave most vector cores idle.
    const int64_t headsPerKvHead = numQueryHeads / numKvHeads;
    const int64_t baseTasks = numTokens * numKvHeads;
    const int64_t desiredTiles = std::max<int64_t>(1, coreCount / baseTasks);
    const int64_t headsPerTask = (headsPerKvHead + desiredTiles - 1) / desiredTiles;
    const int64_t taskTilesPerKvHead = (headsPerKvHead + headsPerTask - 1) / headsPerTask;
    const int64_t taskCount = baseTasks * taskTilesPerKvHead;
    const uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(taskCount, coreCount));
    QsaSparseAttentionV310TilingData data;
    data.set_numTokens(numTokens);
    data.set_numQueryHeads(numQueryHeads);
    data.set_numKvHeads(numKvHeads);
    data.set_headsPerTask(headsPerTask);
    data.set_taskTilesPerKvHead(taskTilesPerKvHead);
    data.set_headDim(headDim);
    data.set_cacheBlockSize(cacheBlockSize);
    data.set_cacheHeadDimBlocks(cacheHeadDimBlocks);
    data.set_maxBlocksPerSequence(blockTable.GetDim(1));
    data.set_selectedGroupsWidth(groups.GetDim(1));
    data.set_numRequests(queryStartLoc.GetDim(0) - 1);
    data.set_taskCount(taskCount);
    data.set_tasksPerCore((taskCount + blockDim - 1) / blockDim);
    data.set_scaleQ24(*scaleQ24);

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

IMPL_OP_OPTILING(QsaSparseAttentionV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
