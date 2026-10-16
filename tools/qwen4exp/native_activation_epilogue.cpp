// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr int64_t WIDTH = 640, ROWS_PER_TILE = 4, TILE = WIDTH * ROWS_PER_TILE;
constexpr float FP16_MAX = 65504.0f;

class ActivationEpilogue {
 public:
  __aicore__ inline void Run(GM_ADDR activated, GM_ADDR projected, GM_ADDR output, int64_t rows) {
    activated_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activated));
    projected_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(projected));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    pipe_.InitBuffer(buffer_, 17 * TILE);
    auto gate = buffer_.Get<float>();
    auto up = gate[TILE];
    auto scratch = gate[2 * TILE];
    auto halfUp = buffer_.Get<half>()[6 * TILE];
    auto halfOut = buffer_.Get<half>()[7 * TILE];
    auto mask = buffer_.Get<uint8_t>()[16 * TILE];
    const int64_t tasks = (rows + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
    for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
      const int64_t first = task * ROWS_PER_TILE;
      const uint16_t count = rows - first < ROWS_PER_TILE ? rows - first : ROWS_PER_TILE;
      Duplicate(gate, 0.0f, TILE);
      Duplicate(halfUp, static_cast<half>(0), TILE);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(gate, activated_[first * WIDTH], count * WIDTH);
      DataCopyParams copy{count, static_cast<uint16_t>(WIDTH * sizeof(half) / 32),
                          static_cast<uint16_t>(WIDTH * sizeof(half) / 32), 0};
      DataCopy(halfUp, projected_[first * WIDTH * 2 + WIDTH], copy);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(up, halfUp, RoundMode::CAST_NONE, TILE);
      PipeBarrier<PIPE_V>();
      Mul(gate, gate, up, TILE);
      PipeBarrier<PIPE_V>();
      // Preserve the retained 310P FP32->FP16 saturation and NaN behavior.
      Mins(gate, gate, FP16_MAX, TILE);
      PipeBarrier<PIPE_V>();
      Maxs(gate, gate, -FP16_MAX, TILE);
      PipeBarrier<PIPE_V>();
      Compare(mask, gate, gate, CMPMODE::EQ, TILE);
      Duplicate(scratch, FP16_MAX, TILE);
      PipeBarrier<PIPE_V>();
      Select(gate, mask, gate, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
      PipeBarrier<PIPE_V>();
      Cast(halfOut, gate, RoundMode::CAST_NONE, TILE);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output_[first * WIDTH], halfOut, count * WIDTH);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buffer_;
  GlobalTensor<float> activated_;
  GlobalTensor<half> projected_, output_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_activation_epilogue_v1(GM_ADDR activated, GM_ADDR projected, GM_ADDR output,
                                                                  GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (rows < 1 || rows > 20480) return;
  ActivationEpilogue op;
  op.Run(activated, projected, output, rows);
}
