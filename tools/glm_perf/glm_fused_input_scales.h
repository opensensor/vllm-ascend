// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
namespace GlmFusedInputScales {
using namespace AscendC;
constexpr unsigned BROADCAST_LANES = 8;
// A copy preserves scale bits. Arithmetic starts only at weight * activation.
__aicore__ inline void CopyRows(LocalTensor<float> dst, LocalTensor<float> scales, LocalTensor<uint32_t> indices,
                                unsigned rows, bool contiguous) {
  if (contiguous)
    DataCopy(dst, scales, rows);
  else
    Gather(dst, scales, indices, static_cast<uint32_t>(0), rows);
}
__aicore__ inline void BroadcastRows(LocalTensor<float> dst, LocalTensor<float> temporary, LocalTensor<float> scales,
                                     LocalTensor<uint32_t> indices, unsigned rows, bool contiguous) {
  if (contiguous) {
    Brcb(dst, scales, rows / BROADCAST_LANES, {1, BROADCAST_LANES});
  } else {
    Gather(temporary, scales, indices, static_cast<uint32_t>(0), rows);
    PipeBarrier<PIPE_V>();
    Brcb(dst, temporary, rows / BROADCAST_LANES, {1, BROADCAST_LANES});
  }
}
}  // namespace GlmFusedInputScales
