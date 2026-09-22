#include "qsa_index_cache_update_v310_tiling.h"

#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t QSA_COMPRESS_RATIO = 4;
constexpr int64_t MAX_INDEX_HEAD_DIM = 256;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

    const auto cache = context->GetInputShape(0)->GetStorageShape();
    const auto keys = context->GetInputShape(1)->GetStorageShape();
    const auto queryStartLoc = context->GetInputShape(2)->GetStorageShape();
    const auto slotMapping = context->GetInputShape(3)->GetStorageShape();
    const auto normWeight = context->GetInputShape(4)->GetStorageShape();
    const auto ropeCos = context->GetInputShape(5)->GetStorageShape();
    const auto ropeSin = context->GetInputShape(6)->GetStorageShape();
    OP_CHECK_IF(cache.GetDimNum() != 3, OP_LOGE(context, "cache must be [blocks,rows,D]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keys.GetDimNum() != 2, OP_LOGE(context, "index keys must be [T,D]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryStartLoc.GetDimNum() != 1, OP_LOGE(context, "query start locations must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(slotMapping.GetDimNum() != 1 || slotMapping.GetDim(0) != keys.GetDim(0),
                OP_LOGE(context, "slot mapping must be [T]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDim(2) != keys.GetDim(1), OP_LOGE(context, "cache/key dimensions differ"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normWeight.GetDimNum() != 1 || normWeight.GetDim(0) != keys.GetDim(1),
                OP_LOGE(context, "key norm weight must be [D]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ropeCos.GetDimNum() != 2 || ropeSin.GetDimNum() != 2 ||
                    ropeCos.GetDim(0) != keys.GetDim(0) || ropeSin.GetDim(0) != keys.GetDim(0) ||
                    ropeCos.GetDim(1) != ropeSin.GetDim(1),
                OP_LOGE(context, "rope cos/sin must be matching [T, rotary_dim] tensors"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keys.GetDim(1) <= 0 || keys.GetDim(1) > MAX_INDEX_HEAD_DIM || keys.GetDim(1) % 16 != 0,
                OP_LOGE(context, "index head dimension must be a multiple of 16 up to 256"),
                return ge::GRAPH_FAILED);

    const int64_t *blockSize = context->GetAttrs()->GetInt(0);
    const int64_t *compressRatio = context->GetAttrs()->GetInt(1);
    const int64_t *rotaryDim = context->GetAttrs()->GetInt(2);
    const float *normEps = context->GetAttrs()->GetFloat(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, compressRatio);
    OP_CHECK_NULL_WITH_CONTEXT(context, rotaryDim);
    OP_CHECK_NULL_WITH_CONTEXT(context, normEps);
    OP_CHECK_IF((*blockSize != 64 && *blockSize != 128) || *compressRatio != QSA_COMPRESS_RATIO,
                OP_LOGE(context, "310P QSA cache update requires block_size 64/128 and compression ratio 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*rotaryDim <= 0 || *rotaryDim > keys.GetDim(1) || *rotaryDim % 2 != 0 ||
                    ropeCos.GetDim(1) != *rotaryDim,
                OP_LOGE(context, "rotary_dim must be positive, even, <= D, and match rope inputs"),
                return ge::GRAPH_FAILED);
    const int64_t groupsPerBlock = *blockSize / *compressRatio;
    OP_CHECK_IF(cache.GetDim(1) != groupsPerBlock + *compressRatio - 1,
                OP_LOGE(context, "cache rows must be block_size/ratio plus ratio-1 scratch rows"),
                return ge::GRAPH_FAILED);

    const int64_t numRequests = queryStartLoc.GetDim(0) - 1;
    const uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(
        1, std::min<int64_t>(numRequests, coreCount)));
    QsaIndexCacheUpdateV310TilingData data;
    data.set_numTokens(keys.GetDim(0));
    data.set_headDim(keys.GetDim(1));
    data.set_cacheRowsPerBlock(cache.GetDim(1));
    data.set_groupsPerBlock(groupsPerBlock);
    data.set_blockSize(*blockSize);
    data.set_rotaryDim(*rotaryDim);
    data.set_numRequests(numRequests);
    data.set_requestsPerCore((numRequests + blockDim - 1) / blockDim);
    data.set_normEps(*normEps);

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

IMPL_OP_OPTILING(QsaIndexCacheUpdateV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
