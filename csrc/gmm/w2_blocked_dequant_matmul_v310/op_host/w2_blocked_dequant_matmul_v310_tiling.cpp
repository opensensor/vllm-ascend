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
 * \file w2_blocked_dequant_matmul_v310_tiling.cpp
 * \brief
 */

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "w2_blocked_dequant_matmul_v310_tiling.h"

namespace optiling {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t CODES_INDEX = 1;
constexpr int64_t BLK = 32;

static ge::graphStatus W2BlockedDequantMatmulTilingFunc(gert::TilingContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "aivCoreNum is 0"), return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto codesShapePtr = context->GetInputShape(CODES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, codesShapePtr);

    auto xShape = xShapePtr->GetStorageShape();
    auto codesShape = codesShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 2, OP_LOGE(context, "x must be 2D [T, K]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(codesShape.GetDimNum() != 2, OP_LOGE(context, "codes must be 2D [N, K]"), return ge::GRAPH_FAILED);

    const int64_t T = xShape.GetDim(0);
    const int64_t K = xShape.GetDim(1);
    const int64_t N = codesShape.GetDim(0);
    const int64_t packedK = codesShape.GetDim(1);
    OP_CHECK_IF(packedK <= 0 || K % packedK != 0,
                OP_LOGE(context, "codes.shape[1] must divide x.shape[1] (K)"),
                return ge::GRAPH_FAILED);
    const int64_t codesPerByte = K / packedK;
    OP_CHECK_IF(codesPerByte != 2 && codesPerByte != 4,
                OP_LOGE(context, "packed codes must contain either 2 (W4) or 4 (W2) values per byte"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(T <= 0 || N <= 0 || K <= 0, OP_LOGE(context, "T/N/K must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(N % BLK != 0 || K % BLK != 0, OP_LOGE(context, "N and K must be multiples of 32"),
                return ge::GRAPH_FAILED);

    // Canonical CANN tiling-data pattern (local optiling object + SaveToBuffer),
    // required once workspace > 0 so the RunForWorkspace probe path works.
    W2BlockedDequantMatmulTilingData tilingData;
    tilingData.set_numTokens(T);
    tilingData.set_nDim(N);
    tilingData.set_kDim(K);
    tilingData.set_codesPerByte(codesPerByte);

    // Cube path workspace: [ Wdq (N*K half) ][ yF (alignUp(T,16)*N float) ]
    const int64_t mAligned = (T + 15) / 16 * 16;
    const size_t wdqBytes = static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(uint16_t);
    const size_t yfBytes = static_cast<size_t>(mAligned) * static_cast<size_t>(N) * sizeof(float);
    // GetUserWorkspace(workspace) returns (workspace + GetLibApiWorkSpaceSize()),
    // so the reported size must include that system reserve or the kernel writes
    // run past the allocation (invalid GM address).
    // Per-core de-interleaved x workspace: each core builds its own field-major
    // copy of x[mAligned, K] so the on-chip unpack can store the weight
    // field-major and skip the per-weight-row gather (x is de-interleaved once
    // per core instead).
    const size_t xfmBytes = static_cast<size_t>(coreNum) * static_cast<size_t>(mAligned)
                            * static_cast<size_t>(K) * sizeof(uint16_t);
    const size_t sysRsv = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysRsv + wdqBytes + yfBytes + xfmBytes;

    const int64_t nBlocks = (N + 127) / 128;
    uint32_t blockDim = (nBlocks < static_cast<int64_t>(coreNum)) ? static_cast<uint32_t>(nBlocks) : coreNum;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForW2BlockedDequantMatmul(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

struct W2BlockedDequantMatmulCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

IMPL_OP_OPTILING(W2BlockedDequantMatmulV310)
    .Tiling(W2BlockedDequantMatmulTilingFunc)
    .TilingParse<W2BlockedDequantMatmulCompileInfo>(TilingParseForW2BlockedDequantMatmul);
}  // namespace optiling
