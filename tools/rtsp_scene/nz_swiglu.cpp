// SPDX-License-Identifier: Apache-2.0
// Decode-only NZ gate/up -> FP16 SwiGLU -> per-token INT8 NZ, without ND GM.
#include "kernel_operator.h"

#ifndef RTSP_DENSE_MAX_ROWS
  #define RTSP_DENSE_MAX_ROWS 8
#endif

namespace {
using namespace AscendC;
constexpr uint32_t HIDDEN = 6144, ROWS = 16, HALF_CELL = 16, INT8_CELL = 32;
constexpr uint32_t SCALE_STRIDE = 8;  // One 32-byte scale slot per core/row.
constexpr uint32_t MAX_DECODE_ROWS = RTSP_DENSE_MAX_ROWS;
static_assert(MAX_DECODE_ROWS == 8 || MAX_DECODE_ROWS == ROWS);
constexpr float QUANT_MAX = 127.0f;
class NZActivation {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR scale_output, GM_ADDR config, GM_ADDR debug) {
    const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
    if (rows < 1 || rows > MAX_DECODE_ROWS || GetBlockIdx() >= rows) return;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale_output));
    pipe_.InitBuffer(gate_, HIDDEN * sizeof(half));
    pipe_.InitBuffer(up_, HIDDEN * sizeof(half));
    pipe_.InitBuffer(g_, HIDDEN * sizeof(float));
    pipe_.InitBuffer(u_, HIDDEN * sizeof(float));
    pipe_.InitBuffer(temp_, HIDDEN * sizeof(float));
    pipe_.InitBuffer(result_, HIDDEN * sizeof(half));
    pipe_.InitBuffer(integers_, HIDDEN * sizeof(int32_t));
    pipe_.InitBuffer(codes_, HIDDEN);
    pipe_.InitBuffer(reduction_, 256 * sizeof(float));
    pipe_.InitBuffer(scale_, SCALE_STRIDE * sizeof(float));
    // 310P exposes eight cores. A 16-row decode reuses each core's buffers
    // for its second row, retaining the same one-block NZ row geometry.
    for (uint32_t row = GetBlockIdx(); row < rows; row += GetBlockNum()) ProcessRow(row, config, debug);
  }

 private:
  __aicore__ inline void ProcessRow(uint32_t row, GM_ADDR config, GM_ADDR debug) {
    auto gate = gate_.Get<half>(), up = up_.Get<half>();
    const DataCopyParams read{HIDDEN / HALF_CELL, 1, ROWS - 1, 0};
    DataCopy(gate, x_[row * HALF_CELL], read);
    DataCopy(up, x_[HIDDEN * ROWS + row * HALF_CELL], read);
    PipeBarrier<PIPE_ALL>();
    auto g = g_.Get<float>(), u = u_.Get<float>(), temp = temp_.Get<float>();
    Cast(g, gate, RoundMode::CAST_NONE, HIDDEN);
    Cast(u, up, RoundMode::CAST_NONE, HIDDEN);
    PipeBarrier<PIPE_V>();
    Muls(temp, g, -1.0f, HIDDEN);
    PipeBarrier<PIPE_V>();
    Exp(temp, temp, HIDDEN);
    PipeBarrier<PIPE_V>();
    Adds(temp, temp, 1.0f, HIDDEN);
    PipeBarrier<PIPE_V>();
    Div(g, g, temp, HIDDEN);
    PipeBarrier<PIPE_V>();
    Mul(g, g, u, HIDDEN);
    PipeBarrier<PIPE_V>();
    // Match the model's FP16 activation boundary before dynamic quantization.
    Cast(result_.Get<half>(), g, RoundMode::CAST_NONE, HIDDEN);
    if (reinterpret_cast<__gm__ int64_t*>(config)[1]) {
      GlobalTensor<half> debug_output;
      debug_output.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(debug));
      PipeBarrier<PIPE_ALL>();
      DataCopy(debug_output[row * HIDDEN], result_.Get<half>(), HIDDEN);
      PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_V>();
    Cast(g, result_.Get<half>(), RoundMode::CAST_NONE, HIDDEN);
    PipeBarrier<PIPE_V>();
    Abs(temp, g, HIDDEN);
    PipeBarrier<PIPE_V>();
    ReduceMax(scale_.Get<float>(), temp, reduction_.Get<float>(), HIDDEN, false);
    PipeBarrier<PIPE_V>();
    // Use the same vector FP32 reciprocal multiplication as DynamicQuant.
    Muls(scale_.Get<float>(), scale_.Get<float>(), 1.0f / QUANT_MAX, 1);
    PipeBarrier<PIPE_ALL>();
    const float scale = scale_.Get<float>().GetValue(0);
    Duplicate(scale_.Get<float>(), scale, SCALE_STRIDE);
    // Match CANN's FP16 multiplication and ties-away integer conversion.
    Muls(result_.Get<half>(), result_.Get<half>(), static_cast<half>(scale == 0 ? 0.0f : 1.0f / scale), HIDDEN);
    PipeBarrier<PIPE_V>();
    Cast(g, result_.Get<half>(), RoundMode::CAST_NONE, HIDDEN);
    PipeBarrier<PIPE_V>();
    Cast(integers_.Get<int32_t>(), g, RoundMode::CAST_ROUND, HIDDEN);
    PipeBarrier<PIPE_V>();
    Cast(g, integers_.Get<int32_t>(), RoundMode::CAST_NONE, HIDDEN);
    PipeBarrier<PIPE_V>();
    Cast(result_.Get<half>(), g, RoundMode::CAST_NONE, HIDDEN);
    PipeBarrier<PIPE_V>();
    Cast(codes_.Get<int8_t>(), result_.Get<half>(), RoundMode::CAST_NONE, HIDDEN);
    PipeBarrier<PIPE_ALL>();
    const DataCopyParams write{HIDDEN / INT8_CELL, 1, 0, ROWS - 1};
    DataCopy(y_[row * INT8_CELL], codes_.Get<int8_t>(), write);
    DataCopy(scales_[row * SCALE_STRIDE], scale_.Get<float>(), SCALE_STRIDE);
    PipeBarrier<PIPE_ALL>();
  }

  TPipe pipe_;
  GlobalTensor<half> x_;
  GlobalTensor<int8_t> y_;
  GlobalTensor<float> scales_;
  TBuf<TPosition::VECCALC> gate_, up_, g_, u_, temp_, result_, integers_, codes_, reduction_, scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void rtsp_nz_swiglu(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR config,
                                                     GM_ADDR debug) {
  NZActivation op;
  op.Run(x, y, scale, config, debug);
}
