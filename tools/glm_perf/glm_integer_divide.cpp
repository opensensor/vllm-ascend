// SPDX-License-Identifier: Apache-2.0
// Exact integer metadata division, without AI-CPU FloorDiv dispatch.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t TILE = 64;
template <typename T, int64_t DIVISOR>
__aicore__ inline void Divide(GM_ADDR input, GM_ADDR output, int64_t count) {
  GlobalTensor<T> source, destination;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
  destination.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, TILE * sizeof(T));
  auto local = buffer.Get<T>();
  constexpr int64_t DMA_ELEMENTS = 32 / sizeof(T);
  // Assign whole aligned output blocks to a core. Scalar GM stores from
  // different cores can overwrite neighbours in the same cache line.
  for (int64_t offset = GetBlockIdx() * TILE; offset < count; offset += GetBlockNum() * TILE) {
    const int64_t valid = count - offset < TILE ? count - offset : TILE;
    const int64_t aligned = (valid + DMA_ELEMENTS - 1) / DMA_ELEMENTS * DMA_ELEMENTS;
    for (int64_t i = 0; i < aligned; ++i) {
      T result = 0;
      if (i < valid) {
        const T value = source.GetValue(offset + i);
        const T quotient = value / DIVISOR;
        result = quotient - (value < 0 && value % DIVISOR != 0);
      }
      local.SetValue(i, result);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(destination[offset], local, static_cast<uint32_t>(aligned));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
}
template <typename T>
__aicore__ inline void Dispatch(GM_ADDR input, GM_ADDR output, int64_t count, int64_t divisor) {
  if (divisor == 4)
    Divide<T, 4>(input, output, count);
  else if (divisor == 32)
    Divide<T, 32>(input, output, count);
  else if (divisor == 160)
    Divide<T, 160>(input, output, count);
  else if (divisor == 640)
    Divide<T, 640>(input, output, count);
}
}  // namespace
extern "C" __global__ __aicore__ void glm_integer_divide_v2(GM_ADDR input, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  const auto values = reinterpret_cast<__gm__ int64_t*>(config);
  if (values[2] == 4)
    Dispatch<int32_t>(input, output, values[0], values[1]);
  else if (values[2] == 8)
    Dispatch<int64_t>(input, output, values[0], values[1]);
}
