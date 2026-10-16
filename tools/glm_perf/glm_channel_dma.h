// SPDX-License-Identifier: Apache-2.0
// 310P cross-pipe dependencies. A vector barrier alone does not publish UB
// writes to MTE3, or protect vector reads from an incoming MTE2 overwrite.
#ifndef GLM_CHANNEL_DMA_H
#define GLM_CHANNEL_DMA_H
#include "kernel_operator.h"
namespace GlmChannelDma {
__aicore__ inline void BeforeRead() {
  using namespace AscendC;
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}
template <class T>
__aicore__ inline void CopyIn(const AscendC::LocalTensor<T>& destination, const AscendC::GlobalTensor<T>& source,
                              uint32_t count) {
  using namespace AscendC;
  BeforeRead();
  DataCopy(destination, source, count);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void WriteReady() {
  using namespace AscendC;
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}
__aicore__ inline void Fence() {
  using namespace AscendC;
  PipeBarrier<PIPE_ALL>();
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WriteReady();
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  PipeBarrier<PIPE_ALL>();
}
}  // namespace GlmChannelDma
#endif
