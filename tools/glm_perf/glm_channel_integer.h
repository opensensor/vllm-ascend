// SPDX-License-Identifier: Apache-2.0
// Exact A8 reconstruction before the one INT32 -> FP32 product conversion.
#ifndef GLM_CHANNEL_INTEGER_H
#define GLM_CHANNEL_INTEGER_H
#include "kernel_operator.h"
namespace GlmChannelInteger {
__aicore__ inline void ReconstructA8(AscendC::LocalTensor<int32_t> low, AscendC::LocalTensor<int32_t> high,
                                     AscendC::LocalTensor<int32_t> sums, uint32_t count) {
  using namespace AscendC;
  constexpr uint32_t N = 128;
  // G<=4096: limb dots, scaled limbs, bias and their sum remain exactly
  // representable in INT32 and FP32. This preserves the previous float order.
  Muls(high, high, static_cast<int32_t>(16), count * N);
  Muls(sums, sums, static_cast<int32_t>(8), N);
  PipeBarrier<PIPE_V>();
  Add(low, low, high, count * N);
  PipeBarrier<PIPE_V>();
  if (count == 1) {
    Add(low, low, sums, N);
  } else {
    const BinaryRepeatParams channelBroadcast{1, 1, 1, 16, 16, 0};
    for (uint32_t column = 0; column < N; column += 64)
      Add(low[column], low[column], sums[column], static_cast<uint64_t>(64), count, channelBroadcast);
  }
  PipeBarrier<PIPE_V>();
}
}  // namespace GlmChannelInteger
#endif
