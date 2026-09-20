/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gdn_gating_v310_tiling.cpp
 * \brief
 */

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "gdn_gating_v310_tiling.h"

namespace optiling {

constexpr uint32_t A_INDEX = 0;
constexpr uint32_t NEG_EXP_INDEX = 2;
// Rows per DataCopy block. The only real constraint is that tileRows * H be a
// multiple of 16 elements, which keeps both the fp16 (2B) and fp32 (4B)
// transfers 32-byte aligned. Using a fixed 32 met that for any H but made decode
// pad a 1-token call out to 32 rows -- 32x wasted vector work, which cost more
// than the launches the operator saves. Take the SMALLEST legal value instead:
// H=12 (TP4) -> 4 rows, H=48 (TP1) -> 1 row.
constexpr int64_t ALIGN_ELEMS = 16;

static int64_t GcdI64(int64_t a, int64_t b)
{
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int64_t TileRowsForHeads(int64_t heads)
{
    return ALIGN_ELEMS / GcdI64(heads, ALIGN_ELEMS);
}
// UB is 248 KB; the kernel holds ~6 fp32 buffers of TILE_ROWS * H.
constexpr int64_t MAX_HEADS = 256;

static ge::graphStatus GdnGatingTilingFunc(gert::TilingContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "aivCoreNum is 0"), return ge::GRAPH_FAILED);

    auto aShapePtr = context->GetInputShape(A_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShapePtr);
    auto negExpShapePtr = context->GetInputShape(NEG_EXP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, negExpShapePtr);

    auto aShape = aShapePtr->GetStorageShape();
    auto negExpShape = negExpShapePtr->GetStorageShape();
    OP_CHECK_IF(aShape.GetDimNum() != 2, OP_LOGE(context, "a must be 2D [T, H]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(negExpShape.GetDimNum() != 2, OP_LOGE(context, "negExpALogTiled must be 2D [TILE_ROWS, H]"),
                return ge::GRAPH_FAILED);

    const int64_t T = aShape.GetDim(0);
    const int64_t H = aShape.GetDim(1);
    OP_CHECK_IF(T <= 0 || H <= 0, OP_LOGE(context, "T/H must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(H > MAX_HEADS, OP_LOGE(context, "H exceeds MAX_HEADS"), return ge::GRAPH_FAILED);
    const int64_t tileRows = TileRowsForHeads(H);
    OP_CHECK_IF(negExpShape.GetDim(0) != tileRows || negExpShape.GetDim(1) != H,
                OP_LOGE(context, "negExpALogTiled must be [tileRows, H]"), return ge::GRAPH_FAILED);

    const float *betaAttr = context->GetAttrs()->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaAttr);
    const float beta = *betaAttr;
    OP_CHECK_IF(beta <= 0.0f, OP_LOGE(context, "beta must be positive"), return ge::GRAPH_FAILED);

    const int64_t tileCount = (T + tileRows - 1) / tileRows;  // last tile may be short
    uint32_t blockDim = (tileCount < static_cast<int64_t>(coreNum)) ? static_cast<uint32_t>(tileCount) : coreNum;
    if (blockDim == 0) {
        blockDim = 1;
    }
    const int64_t tilesPerCore = (tileCount + blockDim - 1) / blockDim;

    GdnGatingTilingData tilingData;
    tilingData.set_numRows(T);
    tilingData.set_numHeads(H);
    tilingData.set_tileRows(tileRows);
    tilingData.set_tilesPerCore(tilesPerCore);
    tilingData.set_tileCount(tileCount);
    tilingData.set_beta(beta);
    tilingData.set_invBeta(1.0f / beta);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGdnGating(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

struct GdnGatingCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

IMPL_OP_OPTILING(GdnGatingV310).Tiling(GdnGatingTilingFunc).TilingParse<GdnGatingCompileInfo>(TilingParseForGdnGating);
}  // namespace optiling
