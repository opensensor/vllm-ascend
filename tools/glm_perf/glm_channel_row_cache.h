// SPDX-License-Identifier: Apache-2.0
// Prepare route addresses once per row block, outside every K/readback loop.
#ifndef GLM_CHANNEL_ROW_CACHE_H
#define GLM_CHANNEL_ROW_CACHE_H
#include "kernel_operator.h"
namespace GlmChannelRowCache {
__aicore__ inline void Prepare(AscendC::GlobalTensor<int64_t> order, AscendC::LocalTensor<int32_t> cache, int64_t row,
                               uint32_t count, int64_t topK) {
  for (uint32_t m = 0; m < count; ++m) {
    const int64_t position = order.GetValue(row + m);
    int64_t token;
    if (topK == 8)
      token = position >> 3;
    else if (topK == 4)
      token = position >> 2;
    else if (topK == 2)
      token = position >> 1;
    else if (topK == 1)
      token = position;
    else
      token = position / topK;
    cache.SetValue(m, static_cast<int32_t>(token));
  }
}
}  // namespace GlmChannelRowCache
#endif
