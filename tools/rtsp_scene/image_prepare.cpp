// SPDX-License-Identifier: Apache-2.0
// Exact fixed-point vertical bicubic resize and FP16 normalization table lookup.
#include "kernel_operator.h"
#ifndef RTSP_IMAGE_WIDTH
  #define RTSP_IMAGE_WIDTH 320
#endif
#ifndef RTSP_IMAGE_TAPS
  #define RTSP_IMAGE_TAPS 4
#endif
#ifndef RTSP_IMAGE_COEFFICIENT_BITS
  #define RTSP_IMAGE_COEFFICIENT_BITS 14
#endif
namespace {
using namespace AscendC;
constexpr uint32_t WIDTH = RTSP_IMAGE_WIDTH, CHANNELS = 3, ROW = WIDTH * CHANNELS;
constexpr uint32_t TAPS = RTSP_IMAGE_TAPS, LUT_SIZE = 256;
constexpr uint32_t COEFFICIENT_BITS = RTSP_IMAGE_COEFFICIENT_BITS;
constexpr float ROUNDING = static_cast<float>(1 << (COEFFICIENT_BITS - 1));
constexpr float INVERSE_SHIFT = 1.0f / (1 << COEFFICIENT_BITS);
constexpr float FLOAT_ELEMENT_BYTES = 4.0f;
class ImagePrepare {
 public:
  __aicore__ inline void Run(GM_ADDR rgb, GM_ADDR weights, GM_ADDR source_rows, GM_ADDR indices, GM_ADDR lookup,
                             GM_ADDR output, GM_ADDR config) {
    const auto dimensions = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t frames = dimensions[0], height = dimensions[1], target = dimensions[2];
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(rgb));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(source_rows));
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indices));
    lut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lookup));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    pipe_.InitBuffer(bytes_, ROW);
    pipe_.InitBuffer(halves_, ROW * sizeof(half));
    pipe_.InitBuffer(values_, ROW * sizeof(float));
    pipe_.InitBuffer(sum_, ROW * sizeof(float));
    pipe_.InitBuffer(indices_buf_, CHANNELS * WIDTH * sizeof(uint32_t));
    pipe_.InitBuffer(channel_, WIDTH * sizeof(float));
    pipe_.InitBuffer(offsets_, WIDTH * sizeof(uint32_t));
    pipe_.InitBuffer(integer_, WIDTH * sizeof(int32_t));
    pipe_.InitBuffer(table_, CHANNELS * LUT_SIZE * sizeof(float));
    pipe_.InitBuffer(result_, WIDTH * sizeof(half));
    DataCopy(indices_buf_.Get<uint32_t>(), indices_, CHANNELS * WIDTH);
    DataCopy(table_.Get<float>(), lut_, CHANNELS * LUT_SIZE);
    PipeBarrier<PIPE_ALL>();
    for (int64_t row = GetBlockIdx(); row < frames * target; row += GetBlockNum()) {
      const int64_t frame = row / target, vertical = row % target;
      auto sum = sum_.Get<float>(), values = values_.Get<float>();
      Duplicate(sum, 0.0f, ROW);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t tap = 0; tap < TAPS; ++tap) {
        const float weight = w_.GetValue(vertical * TAPS + tap);
        if (weight == 0) continue;
        const int64_t source_row = rows_.GetValue(vertical * TAPS + tap);
        DataCopy(bytes_.Get<uint8_t>(), x_[(frame * height + source_row) * ROW], ROW);
        PipeBarrier<PIPE_ALL>();
        Cast(halves_.Get<half>(), bytes_.Get<uint8_t>(), RoundMode::CAST_NONE, ROW);
        PipeBarrier<PIPE_V>();
        Cast(values, halves_.Get<half>(), RoundMode::CAST_NONE, ROW);
        PipeBarrier<PIPE_V>();
        Muls(values, values, weight, ROW);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, values, ROW);
        PipeBarrier<PIPE_ALL>();
      }
      Adds(sum, sum, ROUNDING, ROW);
      PipeBarrier<PIPE_V>();
      Muls(sum, sum, INVERSE_SHIFT, ROW);
      PipeBarrier<PIPE_V>();
      for (uint32_t channel = 0; channel < CHANNELS; ++channel) {
        auto values = channel_.Get<float>();
        Gather(values, sum, indices_buf_.Get<uint32_t>()[channel * WIDTH], 0, WIDTH);
        PipeBarrier<PIPE_V>();
        Cast(integer_.Get<int32_t>(), values, RoundMode::CAST_FLOOR, WIDTH);
        PipeBarrier<PIPE_V>();
        Cast(values, integer_.Get<int32_t>(), RoundMode::CAST_NONE, WIDTH);
        PipeBarrier<PIPE_V>();
        Maxs(values, values, 0.0f, WIDTH);
        PipeBarrier<PIPE_V>();
        Mins(values, values, 255.0f, WIDTH);
        PipeBarrier<PIPE_V>();
        Muls(values, values, FLOAT_ELEMENT_BYTES, WIDTH);
        PipeBarrier<PIPE_V>();
        Cast(offsets_.Get<int32_t>(), values, RoundMode::CAST_ROUND, WIDTH);
        PipeBarrier<PIPE_V>();
        Gather(values, table_.Get<float>()[channel * LUT_SIZE], offsets_.Get<uint32_t>(), 0, WIDTH);
        PipeBarrier<PIPE_V>();
        Cast(result_.Get<half>(), values, RoundMode::CAST_NONE, WIDTH);
        PipeBarrier<PIPE_ALL>();
        DataCopy(y_[((frame * CHANNELS + channel) * target + vertical) * WIDTH], result_.Get<half>(), WIDTH);
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

 private:
  TPipe pipe_;
  GlobalTensor<uint8_t> x_;
  GlobalTensor<float> w_, lut_;
  GlobalTensor<int64_t> rows_;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<half> y_;
  TBuf<TPosition::VECCALC> bytes_, halves_, values_, sum_, indices_buf_, channel_, offsets_, integer_, table_, result_;
};
}  // namespace
extern "C" __global__ __aicore__ void rtsp_image_prepare(GM_ADDR rgb, GM_ADDR w, GM_ADDR r, GM_ADDR i, GM_ADDR l,
                                                         GM_ADDR y, GM_ADDR c) {
  ImagePrepare op;
  op.Run(rgb, w, r, i, l, y, c);
}
