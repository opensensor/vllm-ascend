// SPDX-License-Identifier: Apache-2.0
// Experimental source: native compilation and NPU parity are pending.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t HEADS_PER_TILE = 16;
constexpr int64_t TILE = HEAD_DIM * HEADS_PER_TILE;
class Prepare {
 public:
  __aicore__ inline void Run(GM_ADDR rawGate, GM_ADDR rawBeta, GM_ADDR scale,
                            GM_ADDR bias, GM_ADDR lower, GM_ADDR gateOut,
                            GM_ADDR betaOut, GM_ADDR tiling) {
    auto td = reinterpret_cast<__gm__ int64_t*>(tiling);
    const int64_t rows = td[0], heads = td[1];
    gateHalf_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rawGate));
    gateFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rawGate));
    betaHalf_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rawBeta));
    betaFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rawBeta));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    lower_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lower));
    gateOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gateOut));
    betaOut_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(betaOut));
    pipe_.InitBuffer(storage_, 3 * TILE * sizeof(float) + TILE * sizeof(half) + 256);
    auto value = storage_.Get<float>();
    auto work = value[TILE];
    auto ones = value[2 * TILE];
    auto halfValue = storage_.Get<half>()[6 * TILE];
    auto scales = value[3 * TILE + TILE / 2];
    auto beta = scales[HEADS_PER_TILE];
    auto betaHalf = beta[HEADS_PER_TILE].ReinterpretCast<half>();
    const float lowerBound = lower_.GetValue(0);
    const int64_t tasks = rows * heads / HEADS_PER_TILE;
    for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
      const int64_t base = task * TILE;
      const int64_t head = task * HEADS_PER_TILE % heads;
      if (td[2]) {
        DataCopy(value, gateFloat_[base], TILE);
      } else {
        DataCopy(halfValue, gateHalf_[base], TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(value, halfValue, RoundMode::CAST_NONE, TILE);
      }
      DataCopy(work, bias_[head * HEAD_DIM], TILE);
      DataCopy(scales, scale_[head], HEADS_PER_TILE);
      PipeBarrier<PIPE_ALL>();
      Add(value, value, work, TILE);
      PipeBarrier<PIPE_V>();
      for (int64_t h = 0; h < HEADS_PER_TILE; ++h) {
        Muls(value[h * HEAD_DIM], value[h * HEAD_DIM], scales.GetValue(h), HEAD_DIM);
      }
      PipeBarrier<PIPE_ALL>();
      Sigmoid(value, work, ones, TILE);
      Muls(value, value, lowerBound, TILE);
      PipeBarrier<PIPE_ALL>();
      DataCopy(gateOut_[base], value, TILE);
      if (td[3]) {
        DataCopy(beta, betaFloat_[task * HEADS_PER_TILE], HEADS_PER_TILE);
      } else {
        DataCopy(betaHalf, betaHalf_[task * HEADS_PER_TILE], HEADS_PER_TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(beta, betaHalf, RoundMode::CAST_NONE, HEADS_PER_TILE);
      }
      PipeBarrier<PIPE_ALL>();
      Sigmoid(beta, work, ones, HEADS_PER_TILE);
      Cast(betaHalf, beta, RoundMode::CAST_NONE, HEADS_PER_TILE);
      PipeBarrier<PIPE_ALL>();
      DataCopy(betaOut_[task * HEADS_PER_TILE], betaHalf, HEADS_PER_TILE);
      PipeBarrier<PIPE_ALL>();
    }
  }
 private:
  __aicore__ inline void Sigmoid(LocalTensor<float> value, LocalTensor<float> work,
                                LocalTensor<float> ones, int64_t count) {
    Muls(work, value, -1.0f, count);
    PipeBarrier<PIPE_V>();
    Exp(work, work, count);
    PipeBarrier<PIPE_V>();
    Adds(work, work, 1.0f, count);
    Duplicate(ones, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Div(value, ones, work, count);
    PipeBarrier<PIPE_V>();
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> gateHalf_, betaHalf_, betaOut_;
  GlobalTensor<float> gateFloat_, betaFloat_, scale_, bias_, lower_, gateOut_;
};
}
extern "C" __global__ __aicore__ void glm_kda_gate_beta_v1(
    GM_ADDR rawGate, GM_ADDR rawBeta, GM_ADDR scale, GM_ADDR bias, GM_ADDR lower,
    GM_ADDR gateOut, GM_ADDR betaOut, GM_ADDR tiling) {
  AscendC::InitSocState();
  Prepare op;
  op.Run(rawGate, rawBeta, scale, bias, lower, gateOut, betaOut, tiling);
}
