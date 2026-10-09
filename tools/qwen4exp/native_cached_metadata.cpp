// SPDX-License-Identifier: Apache-2.0
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"

namespace {
template <uint32_t N>
__aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                           GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y,
                           __gm__ const QwenW4A8Int4KernelTilingData* config) {
  native_int4::Schedule<32, N, false, false, false, true> op;
  op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, config);
  op.Process();
}
}  // namespace
extern "C" __global__ __aicore__ void qwen_cached_metadata_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                              GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                              GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y,
                                                              GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(tiling);
  // Match the existing M=32 grouped prefill schedule and its N selection.
  // Never silently replace decode, routed IDs, large M tiles or older lanes.
  if (td->routed || td->metadataLanes != 8 || td->numExperts != 128 || td->numRows <= 128 || td->numRows > 25600 ||
      td->kDim > 2560 || td->kDim % 128 || td->nDim % 64) {
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid metadata-cache specialization"); });
    return;
  }
  if (td->nDim % (AscendC::GetBlockNum() * 160) == 0)
    Run<160>(low, high, xs, sums, codes, scale, offset, weightSum, ends, y, td);
  else if (td->nDim % (AscendC::GetBlockNum() * 80) == 0)
    Run<80>(low, high, xs, sums, codes, scale, offset, weightSum, ends, y, td);
  else
    Run<64>(low, high, xs, sums, codes, scale, offset, weightSum, ends, y, td);
}
