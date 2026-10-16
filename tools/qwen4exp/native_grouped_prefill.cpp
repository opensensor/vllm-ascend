// SPDX-License-Identifier: Apache-2.0
// Reuse the qualified signed-INT4 arithmetic and ordered FP32 epilogue.
// This entry is prefill-only; routed decode remains on the installed operator.
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"

namespace {
using namespace AscendC;
constexpr int64_t LARGE_ROWS_PER_EXPERT = 256;
constexpr uint32_t WIDE_COLUMNS = 160;

template <bool LOCAL_ROWS>
__aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                           GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y, GM_ADDR descriptor) {
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  // The final group end counts only this rank's work. Reading it on-device
  // avoids a host synchronization and excludes padded routes to peer experts.
  GlobalTensor<int64_t> boundaries;
  boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
  const int64_t rows = LOCAL_ROWS ? boundaries.GetValue(td->numExperts - 1) : td->numRows;
  if (rows > td->numExperts * LARGE_ROWS_PER_EXPERT) {
    native_int4::Schedule<128> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
    op.Process();
  } else {
    native_int4::Schedule<32, WIDE_COLUMNS> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
    op.Process();
  }
}
}  // namespace

extern "C" __global__ __aicore__ void qwen_grouped_local_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                                    GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                                    GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y,
                                                                    GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  Run<true>(low, high, xs, sums, codes, scale, offset, weightSum, ends, y, descriptor);
}

extern "C" __global__ __aicore__ void qwen_grouped_global_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs,
                                                                     GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                                                                     GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends,
                                                                     GM_ADDR y, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  Run<false>(low, high, xs, sums, codes, scale, offset, weightSum, ends, y, descriptor);
}

extern "C" __global__ __aicore__ void qwen_grouped_cached_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs,
                                                                     GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                                                                     GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends,
                                                                     GM_ADDR y, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  native_int4::Schedule<32, 160, false, false, false, true> operation;
  operation.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
  operation.Process();
}

extern "C" __global__ __aicore__ void qwen_grouped_paired_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs,
                                                                     GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                                                                     GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends,
                                                                     GM_ADDR y, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  native_int4::Schedule<32, 160, false, false, false, false, 0, true> operation;
  operation.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
  operation.Process();
}
