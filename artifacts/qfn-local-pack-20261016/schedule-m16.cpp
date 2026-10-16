// SPDX-License-Identifier: Apache-2.0
// Compile only with frozen qualified headers, never newer local schedules.
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"

namespace {
using namespace AscendC;
constexpr int64_t LARGE_ROWS_PER_EXPERT = 64;
constexpr uint32_t MODEL_COLUMNS = 160;

template <int MODE>
__aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                           GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y, GM_ADDR descriptor) {
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  // Only the audited model's grouped prefill shapes; never routed decode.
  if (td->routed != 0 || td->metadataLanes != 8 || td->broadcastFactor != 1 || td->numRows <= 128 ||
      td->numExperts != 128 || !((td->nDim == 1280 && td->kDim == 2560) || (td->nDim == 2560 && td->kDim == 640)))
    return;
  int64_t rows = td->numRows;
  if constexpr (MODE == 2) {
    GlobalTensor<int64_t> boundaries;
    boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    rows = boundaries.GetValue(td->numExperts - 1);
  }
  if (MODE != 1 && rows > td->numExperts * LARGE_ROWS_PER_EXPERT) {
    native_int4::Schedule<128> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
    op.Process();
  } else {
    native_int4::Schedule<16, MODEL_COLUMNS> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
    op.Process();
  }
}
}  // namespace

#define PREFILL_ENTRY(NAME, MODE)                                                                                \
  extern "C" __global__ __aicore__ void NAME(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, \
                                             GM_ADDR scale, GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends,     \
                                             GM_ADDR y, GM_ADDR descriptor) {                                    \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);                                                                \
    AscendC::InitSocState();                                                                                     \
    Run<MODE>(low, high, xs, sums, codes, scale, offset, weightSum, ends, y, descriptor);                        \
  }

PREFILL_ENTRY(qwen_prefill_baseline_m16_v1, 0)
PREFILL_ENTRY(qwen_prefill_small_m16_v1, 1)
PREFILL_ENTRY(qwen_prefill_local_count_m16_v1, 2)
