// SPDX-License-Identifier: Apache-2.0
// Local-only FP32 input + SDMA-populated local inbox. No imported peer reads.
#include "kernel_operator.h"

#if SIDECARD_PAIR_ROWS == 1
  #define SIDECARD_PAIR_ADD_ENTRY sidecard_pair_add_r1_v1
#elif SIDECARD_PAIR_ROWS == 4
  #define SIDECARD_PAIR_ADD_ENTRY sidecard_pair_add_r4_v1
#elif SIDECARD_PAIR_ROWS == 32
  #define SIDECARD_PAIR_ADD_ENTRY sidecard_pair_add_r32_v1
#else
  #error "compile one private kernel for SIDECARD_PAIR_ROWS=1,4,32"
#endif

namespace {
using namespace AscendC;
constexpr uint64_t TILE = 512;
constexpr uint64_t ELEMENTS = SIDECARD_PAIR_ROWS * 4096;
static_assert(ELEMENTS % TILE == 0, "fixed shapes require complete DMA/vector tiles");

class LocalAdd {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR inbox, GM_ADDR output)
  {
    GlobalTensor<float> a, b, y;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inbox));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(input_, TILE * sizeof(float));
    pipe_.InitBuffer(inbox_, TILE * sizeof(float));
    pipe_.InitBuffer(output_, TILE * sizeof(float));
    auto local_a = input_.Get<float>();
    auto local_b = inbox_.Get<float>();
    auto local_y = output_.Get<float>();
    for (uint64_t offset = GetBlockIdx() * TILE; offset < ELEMENTS; offset += GetBlockNum() * TILE) {
      DataCopy(local_a, a[offset], TILE);
      DataCopy(local_b, b[offset], TILE);
      PipeBarrier<PIPE_ALL>();
      Add(local_y, local_a, local_b, TILE);
      PipeBarrier<PIPE_ALL>();
      DataCopy(y[offset], local_y, TILE);
      // Complete the store before reusing output UB for another tile.
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, inbox_, output_;
};
}  // namespace

extern "C" __global__ __aicore__ void SIDECARD_PAIR_ADD_ENTRY(GM_ADDR input, GM_ADDR inbox, GM_ADDR output)
{
  AscendC::InitSocState();
  LocalAdd add;
  add.Run(input, inbox, output);
}
