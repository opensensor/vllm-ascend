// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr int64_t WIDTH = 2560, STREAMS = 4, TILE = WIDTH, MAX_ROWS = 2048;
constexpr float FP16_MAX = 65504.0f;

// Match the existing 310P saturating FP32->FP16 conversion, including NaN.
__aicore__ inline void Narrow(LocalTensor<half> out, LocalTensor<float> value, LocalTensor<float> scratch,
                              LocalTensor<uint8_t> mask) {
  Mins(value, value, FP16_MAX, TILE);
  PipeBarrier<PIPE_V>();
  Maxs(value, value, -FP16_MAX, TILE);
  PipeBarrier<PIPE_V>();
  Compare(mask, value, value, CMPMODE::EQ, TILE);
  Duplicate(scratch, FP16_MAX, TILE);
  PipeBarrier<PIPE_V>();
  Select(value, mask, value, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
  PipeBarrier<PIPE_V>();
  Cast(out, value, RoundMode::CAST_NONE, TILE);
}

class Epilogues {
 public:
  __aicore__ inline void Affine(GM_ADDR input, GM_ADDR affine, GM_ADDR full, GM_ADDR narrow, int64_t rows) {
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    weight_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(affine));
    full_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(full));
    halfOut_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(narrow));
    Init();
    auto x = buffer_.Get<float>();
    auto w = x[TILE];
    auto scratch = x[2 * TILE];
    auto out = buffer_.Get<half>()[6 * TILE];
    auto mask = buffer_.Get<uint8_t>()[16 * TILE];
    for (int64_t task = GetBlockIdx(); task < rows * STREAMS * WIDTH / TILE; task += GetBlockNum()) {
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      DataCopy(x, x_[task * TILE], TILE);
      DataCopy(w, weight_[(task * TILE) % (STREAMS * WIDTH)], TILE);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Mul(x, x, w, TILE);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(full_[task * TILE], x, TILE);
      // Keep the exact FP32 product before clamping the narrowing operand.
      SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
      Narrow(out, x, scratch, mask);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(halfOut_[task * TILE], out, TILE);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    Finish();
  }

  __aicore__ inline void Mean(GM_ADDR input, GM_ADDR gate, GM_ADDR output, int64_t rows) {
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    gate_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gate));
    halfOut_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    Init();
    auto x = buffer_.Get<float>();
    auto g = x[TILE];
    auto sum = x[2 * TILE];
    auto h = buffer_.Get<half>()[6 * TILE];
    // The next task's MTE2 can run ahead of its MTE3_V wait. Its gate
    // buffer must never alias the previous task's pending output store.
    auto outputHalf = h[TILE];
    auto mask = buffer_.Get<uint8_t>()[16 * TILE];
    for (int64_t task = GetBlockIdx(); task < rows * WIDTH / TILE; task += GetBlockNum()) {
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      Duplicate(sum, 0.0f, TILE);
      for (int64_t stream = 0; stream < STREAMS; ++stream) {
        const int64_t offset = (task / (WIDTH / TILE) * STREAMS + stream) * WIDTH + task % (WIDTH / TILE) * TILE;
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(x, x_[offset], TILE);
        DataCopy(h, gate_[offset], TILE);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(g, h, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_V>();
        Mul(x, x, g, TILE);
        PipeBarrier<PIPE_V>();
        // Preserve the qualified left-to-right FP32 reduction, including +0.
        Add(sum, sum, x, TILE);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      }
      PipeBarrier<PIPE_V>();
      Muls(sum, sum, 0.25f, TILE);
      PipeBarrier<PIPE_V>();
      Narrow(outputHalf, sum, g, mask);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(halfOut_[task * TILE], outputHalf, TILE);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    Finish();
  }

 private:
  __aicore__ inline void Init() {
    pipe_.InitBuffer(buffer_, 17 * TILE);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  __aicore__ inline void Finish() {
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buffer_;
  GlobalTensor<float> x_, weight_, full_;
  GlobalTensor<half> gate_, halfOut_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_hc_affine_v1(GM_ADDR input, GM_ADDR affine, GM_ADDR full, GM_ADDR narrow,
                                                        GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (rows <= 0 || rows > MAX_ROWS) return;
  Epilogues op;
  op.Affine(input, affine, full, narrow, rows);
}

extern "C" __global__ __aicore__ void qwen_hc_mean_epilogue_v1(GM_ADDR input, GM_ADDR gate, GM_ADDR output,
                                                               GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (rows <= 0 || rows > MAX_ROWS) return;
  Epilogues op;
  op.Mean(input, gate, output, rows);
}
