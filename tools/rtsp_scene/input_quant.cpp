// SPDX-License-Identifier: Apache-2.0
// FP16 per-token dynamic quantization directly into the narrow matrix's NZ layout.
#include "kernel_operator.h"

#ifndef RTSP_QUANT_INNER
  #define RTSP_QUANT_INNER 4096
#endif

namespace {
using namespace AscendC;
constexpr uint32_t INNER = RTSP_QUANT_INNER, ROW_BLOCK = 16, CELL = 32, SCALE_STRIDE = 8;
constexpr uint32_t MAX_ROWS = 8;
constexpr float QUANT_MAX = 127.0f;
static_assert(INNER == 2048 || INNER == 4096 || INNER == 6144);

class InputQuant {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR scale_output, GM_ADDR config) {
    const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
    const uint32_t row = GetBlockIdx();
    if (rows < 1 || rows > MAX_ROWS || row >= rows) return;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale_output));
    pipe_.InitBuffer(values_, INNER * sizeof(half));
    pipe_.InitBuffer(floats_, INNER * sizeof(float));
    pipe_.InitBuffer(integers_, INNER * sizeof(int32_t));
    pipe_.InitBuffer(codes_, INNER);
    pipe_.InitBuffer(reduction_, 256 * sizeof(float));
    pipe_.InitBuffer(scale_, SCALE_STRIDE * sizeof(float));
    auto values = values_.Get<half>();
    auto floats = floats_.Get<float>();
    DataCopy(values, x_[row * INNER], INNER);
    PipeBarrier<PIPE_ALL>();
    Cast(floats, values, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Abs(floats, floats, INNER);
    PipeBarrier<PIPE_V>();
    ReduceMax(scale_.Get<float>(), floats, reduction_.Get<float>(), INNER, false);
    PipeBarrier<PIPE_V>();
    Muls(scale_.Get<float>(), scale_.Get<float>(), 1.0f / QUANT_MAX, 1);
    PipeBarrier<PIPE_ALL>();
    const float scale = scale_.Get<float>().GetValue(0);
    Duplicate(scale_.Get<float>(), scale, SCALE_STRIDE);
    // Retain CANN's FP16 multiplication and ties-away integer rounding.
    Muls(values, values, static_cast<half>(scale == 0 ? 0.0f : 1.0f / scale), INNER);
    PipeBarrier<PIPE_V>();
    Cast(floats, values, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Cast(integers_.Get<int32_t>(), floats, RoundMode::CAST_ROUND, INNER);
    PipeBarrier<PIPE_V>();
    Cast(floats, integers_.Get<int32_t>(), RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Cast(values, floats, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Cast(codes_.Get<int8_t>(), values, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_ALL>();
    const DataCopyParams write{INNER / CELL, 1, 0, ROW_BLOCK - 1};
    DataCopy(y_[row * CELL], codes_.Get<int8_t>(), write);
    DataCopy(scales_[row * SCALE_STRIDE], scale_.Get<float>(), SCALE_STRIDE);
    PipeBarrier<PIPE_ALL>();
  }

 private:
  TPipe pipe_;
  GlobalTensor<half> x_;
  GlobalTensor<int8_t> y_;
  GlobalTensor<float> scales_;
  TBuf<TPosition::VECCALC> values_, floats_, integers_, codes_, reduction_, scale_;
};
}  // namespace

extern "C" __global__ __aicore__ void rtsp_input_quant(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR config) {
  InputQuant op;
  op.Run(x, y, scale, config);
}
