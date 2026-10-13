// SPDX-License-Identifier: Apache-2.0
#include "qwen_streaming_next_projection.h"

namespace {
constexpr int64_t MAX_STREAMING_ROWS = 25600;
constexpr int64_t MAX_STREAMING_EXPERTS = 128;

__aicore__ inline bool ValidConfiguration(__gm__ const int64_t* c) {
  return c[0] >= 1 && c[0] <= MAX_STREAMING_ROWS && c[1] >= 1 && c[1] <= MAX_STREAMING_EXPERTS &&
         c[2] >= qwen_streaming_next::N && c[2] <= qwen_streaming_next::MAX_K && c[2] % qwen_streaming_next::N == 0 &&
         c[3] >= qwen_streaming_next::GROUP && c[3] <= qwen_streaming_next::MAX_K &&
         c[3] % qwen_streaming_next::GROUP == 0 && c[4] == qwen_streaming_next::LANES && c[5] == 0 && c[6] == 1 &&
         AscendC::GetBlockNum() == qwen_streaming_next::BLOCKS;
}
}  // namespace

extern "C" __global__ __aicore__ void qwen_streaming_projection_v2(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                                   GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                                   GM_ADDR weightSum, GM_ADDR ends, GM_ADDR output,
                                                                   GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  const auto c = reinterpret_cast<__gm__ const int64_t*>(config);
  if (!ValidConfiguration(c)) {
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported grouped streaming geometry"); });
    return;
  }
  qwen_streaming_next::Projection op;
  op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, output, c);
  op.Process();
}

extern "C" __global__ __aicore__ void qwen_streaming_columns_v2(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                                GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                                GM_ADDR weightSum, GM_ADDR ends, GM_ADDR output,
                                                                GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  const auto c = reinterpret_cast<__gm__ const int64_t*>(config);
  if (!ValidConfiguration(c) || c[7] < 0 || c[7] >= c[2] / qwen_streaming_next::N || c[8] < 1 ||
      c[8] > qwen_streaming_next::BLOCKS || c[8] > c[2] / qwen_streaming_next::N - c[7]) {
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported streaming column window"); });
    return;
  }
  qwen_streaming_next::Projection op;
  op.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, output, c);
  op.SetColumns(c[7], c[8]);
  op.Process();
}
