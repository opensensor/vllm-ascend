// SPDX-License-Identifier: Apache-2.0
// Fuse selected-slot FP32 state gather/mask/transpose and inverse scatter.
#include "kernel_operator.h"

namespace {
constexpr int64_t HEAD_DIM = 128, TILE = 8, TILES = HEAD_DIM / TILE;
template <bool SCATTER>
__aicore__ inline void StateLayout(GM_ADDR cache, GM_ADDR slots, GM_ADDR initialized, GM_ADDR packed, GM_ADDR config) {
  using namespace AscendC;
  InitSocState();
  auto c = reinterpret_cast<__gm__ int64_t*>(config);
  const int64_t sequences = c[0], heads = c[1];
  GlobalTensor<float> states, kernelState;
  GlobalTensor<int32_t> indices;
  GlobalTensor<uint8_t> valid;
  states.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cache));
  kernelState.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packed));
  indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slots));
  valid.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(initialized));
  TPipe pipe;
  TBuf<TPosition::VECCALC> input, output;
  pipe.InitBuffer(input, TILE * TILE * sizeof(float));
  pipe.InitBuffer(output, TILE * TILE * sizeof(float));
  auto in = input.Get<float>(), out = output.Get<float>();
  for (int64_t task = GetBlockIdx(); task < sequences * heads * TILES * TILES; task += GetBlockNum()) {
    const int64_t column = task % TILES * TILE, row = task / TILES % TILES * TILE;
    const int64_t head = task / (TILES * TILES) % heads, seq = task / (TILES * TILES * heads);
    const int64_t slot = indices.GetValue(seq);
    const int64_t cacheBase = (slot * heads + head) * HEAD_DIM * HEAD_DIM;
    const int64_t packedBase = (seq * heads + head) * HEAD_DIM * HEAD_DIM;
    // SCATTER receives a fresh initialized value, including cold-state output.
    // GATHER cold rows never read the old cache, including NaNs in stale slots.
    if (SCATTER || valid.GetValue(seq) != 0) {
      for (int64_t r = 0; r < TILE; ++r) {
        if constexpr (SCATTER)
          DataCopy(in[r * TILE], kernelState[packedBase + (row + r) * HEAD_DIM + column], TILE);
        else
          DataCopy(in[r * TILE], states[cacheBase + (row + r) * HEAD_DIM + column], TILE);
      }
      SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      for (int64_t r = 0; r < TILE; ++r)
        for (int64_t col = 0; col < TILE; ++col) out.SetValue(col * TILE + r, in.GetValue(r * TILE + col));
    } else {
      for (int64_t i = 0; i < TILE * TILE; ++i) out.SetValue(i, 0.0f);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    for (int64_t r = 0; r < TILE; ++r) {
      if constexpr (SCATTER)
        DataCopy(states[cacheBase + (column + r) * HEAD_DIM + row], out[r * TILE], TILE);
      else
        DataCopy(kernelState[packedBase + (column + r) * HEAD_DIM + row], out[r * TILE], TILE);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}
}  // namespace
extern "C" __global__ __aicore__ void qwen_state_gather_v1(GM_ADDR cache, GM_ADDR slots, GM_ADDR initialized,
                                                           GM_ADDR packed, GM_ADDR config) {
  StateLayout<false>(cache, slots, initialized, packed, config);
}
extern "C" __global__ __aicore__ void qwen_state_scatter_v1(GM_ADDR cache, GM_ADDR slots, GM_ADDR initialized,
                                                            GM_ADDR packed, GM_ADDR config) {
  StateLayout<true>(cache, slots, initialized, packed, config);
}
