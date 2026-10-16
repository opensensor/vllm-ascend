// SPDX-License-Identifier: Apache-2.0
// CPU-compatible fixed-point 320 -> 352 horizontal RGB byte resize.
#include "kernel_operator.h"
#ifndef RTSP_IMAGE_SOURCE_WIDTH
  #define RTSP_IMAGE_SOURCE_WIDTH 320
#endif
#ifndef RTSP_IMAGE_TARGET_WIDTH
  #define RTSP_IMAGE_TARGET_WIDTH 352
#endif
#ifndef RTSP_IMAGE_TAPS
  #define RTSP_IMAGE_TAPS 4
#endif
#ifndef RTSP_IMAGE_COEFFICIENT_BITS
  #define RTSP_IMAGE_COEFFICIENT_BITS 14
#endif
namespace {
using namespace AscendC;
constexpr uint32_t SOURCE = RTSP_IMAGE_SOURCE_WIDTH * 3, TARGET = RTSP_IMAGE_TARGET_WIDTH * 3;
constexpr uint32_t TAPS = RTSP_IMAGE_TAPS, COEFFICIENT_BITS = RTSP_IMAGE_COEFFICIENT_BITS;
constexpr float ROUNDING = 1 << (COEFFICIENT_BITS - 1), INVERSE_SHIFT = 1.0f / (1 << COEFFICIENT_BITS);
class HorizontalResize {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR weights, GM_ADDR offsets, GM_ADDR output, GM_ADDR config) {
    const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    o_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
    pipe_.InitBuffer(bytes_, SOURCE);
    pipe_.InitBuffer(half_input_, SOURCE * sizeof(half));
    pipe_.InitBuffer(float_input_, SOURCE * sizeof(float));
    pipe_.InitBuffer(weights_, TAPS * TARGET * sizeof(float));
    pipe_.InitBuffer(offsets_, TAPS * TARGET * sizeof(uint32_t));
    pipe_.InitBuffer(values_, TARGET * sizeof(float));
    pipe_.InitBuffer(sum_, TARGET * sizeof(float));
    pipe_.InitBuffer(integers_, TARGET * sizeof(int32_t));
    pipe_.InitBuffer(halves_, TARGET * sizeof(half));
    pipe_.InitBuffer(result_, TARGET);
    DataCopy(weights_.Get<float>(), w_, TAPS * TARGET);
    DataCopy(offsets_.Get<uint32_t>(), o_, TAPS * TARGET);
    PipeBarrier<PIPE_ALL>();
    for (int64_t row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
      DataCopy(bytes_.Get<uint8_t>(), x_[row * SOURCE], SOURCE);
      PipeBarrier<PIPE_ALL>();
      Cast(half_input_.Get<half>(), bytes_.Get<uint8_t>(), RoundMode::CAST_NONE, SOURCE);
      PipeBarrier<PIPE_V>();
      Cast(float_input_.Get<float>(), half_input_.Get<half>(), RoundMode::CAST_NONE, SOURCE);
      auto sum = sum_.Get<float>(), values = values_.Get<float>();
      Duplicate(sum, 0.0f, TARGET);
      PipeBarrier<PIPE_V>();
      for (uint32_t tap = 0; tap < TAPS; ++tap) {
        Gather(values, float_input_.Get<float>(), offsets_.Get<uint32_t>()[tap * TARGET], 0, TARGET);
        PipeBarrier<PIPE_V>();
        Mul(values, values, weights_.Get<float>()[tap * TARGET], TARGET);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, values, TARGET);
        PipeBarrier<PIPE_V>();
      }
      Adds(sum, sum, ROUNDING, TARGET);
      PipeBarrier<PIPE_V>();
      Muls(sum, sum, INVERSE_SHIFT, TARGET);
      PipeBarrier<PIPE_V>();
      Cast(integers_.Get<int32_t>(), sum, RoundMode::CAST_FLOOR, TARGET);
      PipeBarrier<PIPE_V>();
      Cast(sum, integers_.Get<int32_t>(), RoundMode::CAST_NONE, TARGET);
      PipeBarrier<PIPE_V>();
      Maxs(sum, sum, 0.0f, TARGET);
      PipeBarrier<PIPE_V>();
      Mins(sum, sum, 255.0f, TARGET);
      PipeBarrier<PIPE_V>();
      Cast(halves_.Get<half>(), sum, RoundMode::CAST_NONE, TARGET);
      PipeBarrier<PIPE_V>();
      Cast(result_.Get<uint8_t>(), halves_.Get<half>(), RoundMode::CAST_NONE, TARGET);
      PipeBarrier<PIPE_ALL>();
      DataCopy(y_[row * TARGET], result_.Get<uint8_t>(), TARGET);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  GlobalTensor<uint8_t> x_, y_;
  GlobalTensor<float> w_;
  GlobalTensor<uint32_t> o_;
  TBuf<TPosition::VECCALC> bytes_, half_input_, float_input_, weights_, offsets_, values_, sum_, integers_, halves_,
      result_;
};
}  // namespace
extern "C" __global__ __aicore__ void rtsp_image_horizontal(GM_ADDR x, GM_ADDR w, GM_ADDR o, GM_ADDR y, GM_ADDR c) {
  HorizontalResize op;
  op.Run(x, w, o, y, c);
}
