// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_down_reduce_v310_tiling_data.h"

extern "C" __global__ __aicore__ void qwen_w4_a8_int4_down_reduce_v310(
    GM_ADDR low, GM_ADDR high, GM_ADDR activationScale, GM_ADDR activationSum, GM_ADDR codes, GM_ADDR scale,
    GM_ADDR offset, GM_ADDR weightSum, GM_ADDR routeIds, GM_ADDR routeWeights, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4DownReduceKernelTilingData*>(tiling);
  native_int4::Schedule<16, 320, true, true> op;
  op.Init(low, high, activationScale, activationSum, codes, scale, offset, weightSum, routeIds, routeWeights, y, td);
  op.Process();
}
