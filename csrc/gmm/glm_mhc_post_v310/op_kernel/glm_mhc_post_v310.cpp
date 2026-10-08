// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "mhc_post_geometry.h"
namespace {
using namespace AscendC;
constexpr uint32_t TILE = NsGlmMhcPost::TILE;
constexpr uint32_t STREAMS = NsGlmMhcPost::STREAMS;
constexpr uint32_t FLOAT_TILES = STREAMS + 3;  // sources, x, accumulator, scratch
constexpr uint32_t STORAGE_BYTES = TILE * (FLOAT_TILES * sizeof(float) + 2 * sizeof(half)) + TILE;
constexpr uint32_t COMPARE_LANES = 64;  // 310P FP32 compare requires full 256-byte repeats.
constexpr float FP16_MAX = 65504.0f;
struct MhcPostTiling { int64_t rows, width; };
class MhcPost {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb,
                             GM_ADDR y, const MhcPostTiling& td) {
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    residual_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(residual));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(comb));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    pipe_.InitBuffer(storage_, STORAGE_BYTES);
    auto sources = storage_.Get<float>();
    auto input = sources[STREAMS * TILE];
    auto accumulator = sources[(STREAMS + 1) * TILE];
    auto scratch = sources[(STREAMS + 2) * TILE];
    auto inputHalf = storage_.Get<half>()[FLOAT_TILES * TILE * 2];
    auto roundedHalf = inputHalf[TILE];
    auto numberMask = storage_.Get<uint8_t>()[TILE * (FLOAT_TILES * sizeof(float) + 2 * sizeof(half))];
    const int64_t tiles = (td.width + TILE - 1) / TILE;
    for (int64_t task = GetBlockIdx(); task < td.rows * tiles; task += GetBlockNum()) {
      const int64_t row = task / tiles, col = task % tiles * TILE;
      const uint32_t count = td.width - col < TILE ? td.width - col : TILE;
      // Scalar coefficients are device reads. There is no host synchronization.
      float postValues[STREAMS];
      float combValues[STREAMS * STREAMS];
      for (uint32_t i = 0; i < STREAMS; ++i) {
        postValues[i] = post_.GetValue(row * STREAMS + i);
        for (uint32_t j = 0; j < STREAMS; ++j) {
          combValues[i * STREAMS + j] = comb_.GetValue(row * STREAMS * STREAMS + i * STREAMS + j);
        }
        DataCopy(sources[i * TILE], residual_[(row * STREAMS + i) * td.width + col], count);
      }
      DataCopy(inputHalf, x_[row * td.width + col], count);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(input, inputHalf, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      for (uint32_t j = 0; j < STREAMS; ++j) {
        Muls(accumulator, sources, combValues[j], count);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 1; i < STREAMS; ++i) {
          Axpy(accumulator, sources[i * TILE], combValues[i * STREAMS + j], count);
          PipeBarrier<PIPE_V>();
        }
        Muls(scratch, input, postValues[j], count);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, scratch, count);
        PipeBarrier<PIPE_V>();
        // The qualified 310P torch_npu cast saturates finite overflow and
        // +/-Inf; NaN becomes +65504. Match that observed backend behavior.
        Mins(accumulator, accumulator, FP16_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(accumulator, accumulator, -FP16_MAX, count);
        PipeBarrier<PIPE_V>();
        const uint32_t comparisonCount = (count + COMPARE_LANES - 1) / COMPARE_LANES * COMPARE_LANES;
        if (comparisonCount != count) {
          Duplicate(accumulator[count], 0.0f, comparisonCount - count);
          PipeBarrier<PIPE_V>();
        }
        Compare(numberMask, accumulator, accumulator, CMPMODE::EQ, comparisonCount);
        Duplicate(scratch, FP16_MAX, count);
        PipeBarrier<PIPE_V>();
        Select(accumulator, numberMask, accumulator, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        Cast(roundedHalf, accumulator, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(accumulator, roundedHalf, RoundMode::CAST_NONE, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(output_[(row * STREAMS + j) * td.width + col], accumulator, count);
        // The next output stream overwrites accumulator: await its DMA first.
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      }
      // All vector reads must finish before the next tile's input DMA.
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> x_;
  GlobalTensor<float> residual_, post_, comb_, output_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_mhc_post_v310(
    GM_ADDR x, GM_ADDR residual, GM_ADDR post_mix, GM_ADDR comb_mix,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  auto raw = reinterpret_cast<__gm__ MhcPostTiling*>(tiling);
  const MhcPostTiling td{raw->rows, raw->width};
  MhcPost op;
  op.Run(x, residual, post_mix, comb_mix, y, td);
}
