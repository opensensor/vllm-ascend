// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "qsa_gather_value_nz_v310.h"

// One AI Vector task owns a token/head and writes either value or transposed-key NZ tiles.
extern "C" __global__ __aicore__ void qsa_gather_value_nz_v310(
    GM_ADDR valueCache, GM_ADDR groupIndices, GM_ADDR groupCounts, GM_ADDR tailStarts,
    GM_ADDR tailCounts, GM_ADDR blockTable, GM_ADDR valueNz, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(QsaGatherValueNzV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(QsaGatherValueNzV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsQsaGatherValueNz::QsaGatherValueNzV310 op;
    op.Init(valueCache, groupIndices, groupCounts, tailStarts, tailCounts, blockTable,
            valueNz, &tilingData, &pipe);
    op.Process();
}
