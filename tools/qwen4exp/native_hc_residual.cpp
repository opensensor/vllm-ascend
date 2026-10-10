// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr int64_t STREAMS = 4;
constexpr float FP16_MAX = 65504.0f;
struct Tiling {
  int64_t rows, width;
};

template <int64_t TILE>
class Residual {
 public:
  __aicore__ inline void Run(GM_ADDR hyper, GM_ADDR block, GM_ADDR injection, GM_ADDR output, const Tiling& td) {
    hyper_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(hyper));
    block_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(block));
    injection_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(injection));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    pipe_.InitBuffer(storage_, TILE * (4 * sizeof(float) + 3 * sizeof(half) + 1));
    auto input = storage_.Get<float>();
    auto residual = input[TILE];
    auto product = input[2 * TILE];
    auto nanValue = input[3 * TILE];
    auto inputHalf = storage_.Get<half>()[8 * TILE];
    auto residualHalf = inputHalf[TILE];
    auto outputHalf = inputHalf[2 * TILE];
    auto mask = storage_.Get<uint8_t>()[TILE * (4 * sizeof(float) + 3 * sizeof(half))];
    // Free-buffer ownership: MTE2 may overwrite a half input only after V
    // consumes it; V may overwrite outputHalf only after MTE3 drains it.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    const int64_t tiles = (td.width + TILE - 1) / TILE;
    for (int64_t task = GetBlockIdx(); task < td.rows * tiles; task += GetBlockNum()) {
      const int64_t row = task / tiles, col = task % tiles * TILE;
      const uint32_t count = td.width - col < TILE ? td.width - col : TILE;
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(inputHalf, block_[row * td.width + col], count);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(input, inputHalf, RoundMode::CAST_NONE, count);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      for (int64_t stream = 0; stream < STREAMS; ++stream) {
        const int64_t offset = (row * STREAMS + stream) * td.width + col;
        const float coefficient = injection_.GetValue(row * STREAMS + stream);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(residualHalf, hyper_[offset], count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(residual, residualHalf, RoundMode::CAST_NONE, count);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PipeBarrier<PIPE_V>();
        // Preserve separate FP32 multiplication and addition; no FMA.
        Muls(product, input, coefficient, count);
        PipeBarrier<PIPE_V>();
        Add(residual, residual, product, count);
        PipeBarrier<PIPE_V>();
        // Match the qualified 310P torch FP32 -> FP16 saturating cast,
        // including its observed NaN -> +65504 behavior.
        Mins(residual, residual, FP16_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(residual, residual, -FP16_MAX, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, residual, residual, CMPMODE::EQ, count);
        Duplicate(nanValue, FP16_MAX, count);
        PipeBarrier<PIPE_V>();
        Select(residual, mask, residual, nanValue, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Cast(outputHalf, residual, RoundMode::CAST_NONE, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(output_[offset], outputHalf, count);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      }
    }
    // Balance free tokens even on idle cores and finish the final stores.
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> hyper_, block_, output_;
  GlobalTensor<float> injection_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_hc_residual_v3(GM_ADDR hyper, GM_ADDR block, GM_ADDR injection,
                                                          GM_ADDR output, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  auto raw = reinterpret_cast<__gm__ Tiling*>(tiling);
  const Tiling td{raw->rows, raw->width};
  AscendC::InitSocState();
  if (td.rows <= 6) {
    Residual<512> op;
    op.Run(hyper, block, injection, output, td);
  } else {
    Residual<2560> op;
    op.Run(hyper, block, injection, output, td);
  }
}
