// SPDX-License-Identifier: Apache-2.0
// Offline candidate: generated from the frozen qualified schedule with only
// the grouped output-tail zeroing removed. Never bind as a generic projection.
#include "native_int4_local_output.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"

extern "C" __global__ __aicore__ void qwen_local_down_projection_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                                    GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                                    GM_ADDR weightSum, GM_ADDR ends, GM_ADDR output,
                                                                    GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  constexpr int64_t EXPERTS = 128, OUTPUTS = 2560, INPUTS = 640, MAX_ROWS = 20480, LARGE_ROWS_PER_EXPERT = 64;
  constexpr uint32_t MODEL_COLUMNS = 160;
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  if (td->routed != 0 || td->metadataLanes != 8 || td->broadcastFactor != 1 || td->numRows <= 128 ||
      td->numRows > MAX_ROWS || td->numExperts != EXPERTS || td->nDim != OUTPUTS || td->kDim != INPUTS)
    return;
  AscendC::GlobalTensor<int64_t> boundaries;
  boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
  const int64_t active = boundaries.GetValue(EXPERTS - 1);
  if (active < 0 || active > td->numRows) {
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid local down projection boundary"); });
    return;
  }
  if (active == 0) return;
  if (active > EXPERTS * LARGE_ROWS_PER_EXPERT) {
    native_int4::Schedule<128> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, output, td);
    op.Process();
  } else {
    native_int4::Schedule<32, MODEL_COLUMNS> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, output, td);
    op.Process();
  }
}
