// SPDX-License-Identifier: Apache-2.0
#ifndef SIDECARD_MHC_PRE_V310_H
#define SIDECARD_MHC_PRE_V310_H
#include "kernel_operator.h"
#include "sidecard_mhc_pre_v310_tiling_data.h"
namespace NsSidecardMhcPreV310 {
using namespace AscendC;
constexpr int32_t WIDTH = 4096;
constexpr int32_t STREAMS = 4;
constexpr int32_t FLAT = WIDTH * STREAMS;
constexpr int32_t MIXES = 24;
constexpr int32_t TILE = 1024;
constexpr int32_t REDUCE_WIDTH = 64;
constexpr int32_t SCALAR_LANES = 8;

// One complete row per AIV. 118,816 bytes UB, zero GM workspace, no
// [rows,24,16384] broadcast. Sinkhorn remains a separate qualified operator.
class Kernel {
 public:
  __aicore__ inline void Init(GM_ADDR residual, GM_ADDR fn, GM_ADDR scale, GM_ADDR base, GM_ADDR normWeight,
                              GM_ADDR post, GM_ADDR logits, GM_ADDR hidden, const SidecardMhcPreV310TilingData* tiling,
                              TPipe* pipe) {
    residualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(residual));
    fnGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(fn));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    baseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base));
    normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(normWeight));
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(post));
    logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits));
    hiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(hidden));
    rows_ = tiling->rows;
    blocks_ = tiling->blocks;
    normEps_ = tiling->normEps;
    hcEps_ = tiling->hcEps;
    pipe->InitBuffer(residualBuf_, FLAT * sizeof(float));
    pipe->InitBuffer(fnBuf_, TILE * sizeof(float));
    pipe->InitBuffer(hiddenBuf_, WIDTH * sizeof(float));
    pipe->InitBuffer(workBuf_, WIDTH * sizeof(float));
    pipe->InitBuffer(normBuf_, WIDTH * sizeof(half));
    pipe->InitBuffer(outputBuf_, WIDTH * sizeof(half));
    pipe->InitBuffer(scalarBuf_, SCALAR_LANES * sizeof(float));
  }
  __aicore__ inline void Process() {
    for (int64_t row = GetBlockIdx(); row < rows_; row += blocks_) {
      ProcessRow(row);
    }
  }

 private:
  __aicore__ inline float Sum(LocalTensor<float> values, int32_t count) {
    // Fixed pairwise halving, then a 64-lane FP32 reduction. The caller
    // owns this scratch, so folding it cannot modify residual/weights.
    auto scalar = scalarBuf_.Get<float>();
    int32_t width = count;
    while (width > REDUCE_WIDTH) {
      width /= 2;
      Add(values, values, values[width], width);
      PipeBarrier<PIPE_ALL>();
    }
    WholeReduceSum(scalar, values, REDUCE_WIDTH, 1, 1, 1, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    return scalar.GetValue(0);
  }
  __aicore__ inline float InvRoot(float value) {
    auto scalar = scalarBuf_.Get<float>();
    Duplicate(scalar, value, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    Sqrt(scalar, scalar, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    return 1.0f / scalar.GetValue(0);
  }
  __aicore__ inline float Affine(float value, float scale, float base) {
    // Explicit vector instructions prevent compiler contraction of a scalar
    // multiply/add across the baseline's FP32 rounding boundary.
    auto scalar = scalarBuf_.Get<float>();
    Duplicate(scalar, value, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    Muls(scalar, scalar, scale, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    Adds(scalar, scalar, base, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    return scalar.GetValue(0);
  }
  __aicore__ inline float Sigmoid(float value) {
    auto scalar = scalarBuf_.Get<float>();
    Duplicate(scalar, -value, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    Exp(scalar, scalar, SCALAR_LANES);
    PipeBarrier<PIPE_ALL>();
    return 1.0f / (1.0f + scalar.GetValue(0));
  }
  __aicore__ inline uint32_t RoundBits(uint32_t bits) {
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0) {
      return (bits & 0x80000000U) | 0x7fc00000U;
    }
    return (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
  }
  __aicore__ inline uint32_t RoundedScalar(float value) {
    auto scalar = scalarBuf_.Get<float>();
    scalar.SetValue(0, value);
    PipeBarrier<PIPE_ALL>();
    return RoundBits(scalar.ReinterpretCast<uint32_t>().GetValue(0));
  }
  __aicore__ inline void ProcessRow(int64_t row) {
    auto residual = residualBuf_.Get<float>();
    auto fn = fnBuf_.Get<float>();
    auto hidden = hiddenBuf_.Get<float>();
    auto work = workBuf_.Get<float>();
    auto norm = normBuf_.Get<half>();
    auto output = outputBuf_.Get<half>();
    DataCopy(residual, residualGm_[row * FLAT], FLAT);
    PipeBarrier<PIPE_ALL>();
    float squareSum = 0.0f;
    for (int32_t offset = 0; offset < FLAT; offset += TILE) {
      Mul(work, residual[offset], residual[offset], TILE);
      PipeBarrier<PIPE_ALL>();
      squareSum += Sum(work, TILE);
    }
    const float inverseFlatNorm = InvRoot(Affine(squareSum, 1.0f / FLAT, normEps_));
    float mixes[MIXES];
    for (int32_t mix = 0; mix < MIXES; ++mix) {
      float sum = 0.0f;
      for (int32_t offset = 0; offset < FLAT; offset += TILE) {
        DataCopy(fn, fnGm_[mix * FLAT + offset], TILE);
        PipeBarrier<PIPE_ALL>();
        Mul(work, residual[offset], fn, TILE);
        PipeBarrier<PIPE_ALL>();
        sum += Sum(work, TILE);
      }
      mixes[mix] = sum * inverseFlatNorm;
    }
    float pre[STREAMS];
    for (int32_t stream = 0; stream < STREAMS; ++stream) {
      pre[stream] = Sigmoid(Affine(mixes[stream], scaleGm_.GetValue(0), baseGm_.GetValue(stream))) + hcEps_;
      const float post =
          2.0f * Sigmoid(Affine(mixes[STREAMS + stream], scaleGm_.GetValue(1), baseGm_.GetValue(STREAMS + stream)));
      postGm_.SetValue(row * STREAMS + stream, RoundedScalar(post));
    }
    for (int32_t index = 0; index < STREAMS * STREAMS; ++index) {
      logitsGm_.SetValue(row * STREAMS * STREAMS + index, Affine(mixes[2 * STREAMS + index], scaleGm_.GetValue(2),
                                                                 baseGm_.GetValue(2 * STREAMS + index)));
    }
    Muls(hidden, residual, pre[0], WIDTH);
    PipeBarrier<PIPE_ALL>();
    for (int32_t stream = 1; stream < STREAMS; ++stream) {
      Muls(work, residual[stream * WIDTH], pre[stream], WIDTH);
      PipeBarrier<PIPE_ALL>();
      Add(hidden, hidden, work, WIDTH);
      PipeBarrier<PIPE_ALL>();
    }
    auto bits = hidden.ReinterpretCast<uint32_t>();
    for (int32_t index = 0; index < WIDTH; ++index) {
      bits.SetValue(index, RoundBits(bits.GetValue(index)));
    }
    PipeBarrier<PIPE_ALL>();
    Mul(work, hidden, hidden, WIDTH);
    PipeBarrier<PIPE_ALL>();
    const float inverseHiddenNorm = InvRoot(Affine(Sum(work, WIDTH), 1.0f / WIDTH, normEps_));
    DataCopy(norm, normGm_, WIDTH);
    PipeBarrier<PIPE_ALL>();
    Cast(work, norm, RoundMode::CAST_NONE, WIDTH);
    PipeBarrier<PIPE_ALL>();
    // Two separate FP32 products, never a fused multiply-add.
    Muls(hidden, hidden, inverseHiddenNorm, WIDTH);
    PipeBarrier<PIPE_ALL>();
    Mul(hidden, hidden, work, WIDTH);
    PipeBarrier<PIPE_ALL>();
    Cast(output, hidden, RoundMode::CAST_NONE, WIDTH);
    PipeBarrier<PIPE_ALL>();
    DataCopy(hiddenGm_[row * WIDTH], output, WIDTH);
    PipeBarrier<PIPE_ALL>();
  }
  GlobalTensor<float> residualGm_, fnGm_, scaleGm_, baseGm_, logitsGm_;
  GlobalTensor<uint32_t> postGm_;
  GlobalTensor<half> normGm_, hiddenGm_;
  TBuf<TPosition::VECCALC> residualBuf_, fnBuf_, hiddenBuf_, workBuf_, normBuf_, outputBuf_, scalarBuf_;
  int64_t rows_ = 0;
  uint32_t blocks_ = 0;
  float normEps_ = 0, hcEps_ = 0;
};
}  // namespace NsSidecardMhcPreV310
#endif
