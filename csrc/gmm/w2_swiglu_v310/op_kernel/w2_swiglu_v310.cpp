// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "swiglu_geometry.h"
namespace {
using namespace AscendC;
constexpr uint32_t TILE = NsW2Swiglu::TILE;
constexpr uint32_t STORAGE_BYTES = TILE * (2 * sizeof(half) + 3 * sizeof(float));
constexpr float FP16_MAX = 65504.0f;
struct SwigluTiling { int64_t rows, width; };
class Swiglu {
 public:
  __aicore__ inline void Run(GM_ADDR gateUp, GM_ADDR y, const SwigluTiling& td) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gateUp));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    pipe_.InitBuffer(storage_, STORAGE_BYTES);
    auto gateHalf = storage_.Get<half>();
    auto upHalf = gateHalf[TILE];
    auto gate = storage_.Get<float>()[TILE];
    auto up = storage_.Get<float>()[2 * TILE];
    auto sigmoid = storage_.Get<float>()[3 * TILE];
    const int64_t tiles = (td.width + TILE - 1) / TILE;
    for (int64_t task = GetBlockIdx(); task < td.rows * tiles; task += GetBlockNum()) {
      const int64_t row = task / tiles, col = task % tiles * TILE;
      const uint32_t count = td.width - col < TILE ? td.width - col : TILE;
      DataCopy(gateHalf, input_[row * 2 * td.width + col], count);
      DataCopy(upHalf, input_[row * 2 * td.width + td.width + col], count);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(gate, gateHalf, RoundMode::CAST_NONE, count);
      Cast(up, upHalf, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      Muls(sigmoid, gate, -1.0f, count);
      PipeBarrier<PIPE_V>();
      Exp(sigmoid, sigmoid, count);
      PipeBarrier<PIPE_V>();
      Adds(sigmoid, sigmoid, 1.0f, count);
      PipeBarrier<PIPE_V>();
      Div(sigmoid, gate, sigmoid, count);
      PipeBarrier<PIPE_V>();
      Mul(gate, sigmoid, up, count);
      PipeBarrier<PIPE_V>();
      // The serving torch_npu FP32->FP16 cast saturates finite overflow;
      // AscendC's vector cast produces infinity unless explicitly clamped.
      Mins(gate, gate, FP16_MAX, count);
      PipeBarrier<PIPE_V>();
      Maxs(gate, gate, -FP16_MAX, count);
      PipeBarrier<PIPE_V>();
      Cast(gateHalf, gate, RoundMode::CAST_NONE, count);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output_[row * td.width + col], gateHalf, count);
      // Both vector reads and the output DMA must finish before reusing UB.
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> input_, output_;
};
}  // namespace
extern "C" __global__ __aicore__ void w2_swiglu_v310(
    GM_ADDR gate_up, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  auto raw = reinterpret_cast<__gm__ SwigluTiling*>(tiling);
  const SwigluTiling td{raw->rows, raw->width};
  Swiglu op;
  op.Run(gate_up, y, td);
}
