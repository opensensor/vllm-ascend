// SPDX-License-Identifier: Apache-2.0
// Repeat-stride epilogue shared by standalone and UB-fused native projections.
#ifndef GLM_CHANNEL_BROADCAST_H
#define GLM_CHANNEL_BROADCAST_H
#include "kernel_operator.h"
namespace GlmChannelBroadcast {
__aicore__ inline void ScaleRows(AscendC::LocalTensor<float> low, AscendC::LocalTensor<float> weights,
                                 AscendC::LocalTensor<float> correction, AscendC::LocalTensor<float> sx, uint32_t count,
                                 uint32_t bits) {
  using namespace AscendC;
  constexpr uint32_t N = 128, LANES = 8;
  if (count == 1) {
    const BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
    for (uint32_t m = 0; m < count; ++m) {
      if (bits == 8) Add(low[m * N], low[m * N], correction, N);
      PipeBarrier<PIPE_V>();
      Mul(low[m * N], low[m * N], weights, N);
      PipeBarrier<PIPE_V>();
      Mul(low[m * N], low[m * N], sx[m * LANES], static_cast<uint64_t>(64), N / 64, broadcast);
      PipeBarrier<PIPE_V>();
    }
  } else {
    const BinaryRepeatParams channelBroadcast{1, 1, 1, 16, 16, 0};
    const BinaryRepeatParams rowBroadcast{1, 1, 0, 16, 16, 1};
    for (uint32_t column = 0; column < N; column += 64) {
      if (bits == 8) {
        Add(low[column], low[column], correction[column], static_cast<uint64_t>(64), count, channelBroadcast);
        PipeBarrier<PIPE_V>();
      }
      Mul(low[column], low[column], weights[column], static_cast<uint64_t>(64), count, channelBroadcast);
      PipeBarrier<PIPE_V>();
      Mul(low[column], low[column], sx, static_cast<uint64_t>(64), count, rowBroadcast);
      PipeBarrier<PIPE_V>();
    }
  }
}
}  // namespace GlmChannelBroadcast
#endif
