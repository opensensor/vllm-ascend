// SPDX-License-Identifier: Apache-2.0
// Experimental FP32 WY forward substitution, with U/W retained on chip.
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr int64_t CHUNK = 64, DIM = 128;
constexpr uint32_t MATRIX = CHUNK * DIM;

class FusedWY {
 public:
  __aicore__ inline void Run(GM_ADDR keys, GM_ADDR values, GM_ADDR gram, GM_ADDR cumulative_g, GM_ADDR beta, GM_ADDR w,
                             GM_ADDR u, GM_ADDR config) {
    auto c = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t batch = c[0], tokens = c[1], keyHeads = c[2], valueHeads = c[3];
    const int64_t chunks = tokens / CHUNK, group = valueHeads / keyHeads;
    GlobalTensor<half> k, v, outputW, outputU;
    GlobalTensor<float> inputGram, inputG, inputBeta;
    k.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(keys));
    v.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(values));
    inputGram.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gram));
    inputG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cumulative_g));
    inputBeta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
    outputW.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w));
    outputU.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(u));
    TPipe pipe;
    TBuf<TPosition::VECCALC> ub;
    constexpr uint32_t FLOATS = 2 * MATRIX + CHUNK * CHUNK + 3 * CHUNK + DIM;
    pipe.InitBuffer(ub, FLOATS * sizeof(float) + DIM * sizeof(half));
    auto localU = ub.Get<float>();
    auto localW = localU[MATRIX];
    auto localGram = localW[MATRIX];
    auto localG = localGram[CHUNK * CHUNK];
    auto coefficients = localG[CHUNK];
    auto decay = coefficients[CHUNK];
    auto product = decay[CHUNK];
    auto rowHalf = ub.Get<half>()[FLOATS * 2];
    for (int64_t task = GetBlockIdx(); task < batch * valueHeads * chunks; task += GetBlockNum()) {
      const int64_t chunk = task % chunks;
      const int64_t head = task / chunks % valueHeads;
      const int64_t b = task / (chunks * valueHeads);
      const int64_t keyHead = head / group;
      const int64_t headOffset = (b * valueHeads + head) * tokens + chunk * CHUNK;
      const int64_t keyOffset = ((b * keyHeads + keyHead) * tokens + chunk * CHUNK) * DIM;
      const int64_t gramOffset = ((b * keyHeads + keyHead) * chunks + chunk) * CHUNK * CHUNK;
      DataCopy(localGram, inputGram[gramOffset], CHUNK * CHUNK);
      DataCopy(localG, inputG[headOffset], CHUNK);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      for (int64_t row = 0; row < CHUNK; ++row) {
        const int64_t valueOffset = ((b * tokens + chunk * CHUNK + row) * valueHeads + head) * DIM;
        const float betaRow = inputBeta.GetValue((b * tokens + chunk * CHUNK + row) * valueHeads + head);
        const float gRow = localG.GetValue(row);
        DataCopy(rowHalf, v[valueOffset], DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(localU[row * DIM], rowHalf, RoundMode::CAST_NONE, DIM);
        PipeBarrier<PIPE_V>();
        Muls(localU[row * DIM], localU[row * DIM], betaRow, DIM);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(rowHalf, k[keyOffset + row * DIM], DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(localW[row * DIM], rowHalf, RoundMode::CAST_NONE, DIM);
        // One vector Exp supplies exp(g_i) and exp(g_i - g_j). The final
        // lane holds exp(g_i); unused coefficient lanes are never consumed.
        for (int64_t j = 0; j < CHUNK - 1; ++j)
          decay.SetValue(j, j < row ? gRow - localG.GetValue(j) : -3.402823466e+38F);
        decay.SetValue(CHUNK - 1, gRow);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Exp(decay, decay, CHUNK);
        PipeBarrier<PIPE_V>();
        Muls(coefficients, localGram[row * CHUNK], -betaRow, CHUNK);
        PipeBarrier<PIPE_V>();
        Mul(coefficients, coefficients, decay, CHUNK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float scale = betaRow * decay.GetValue(CHUNK - 1);
        Muls(localW[row * DIM], localW[row * DIM], scale, DIM);
        PipeBarrier<PIPE_V>();
        for (int64_t j = 0; j < row; ++j) {
          const float coefficient = coefficients.GetValue(j);
          Muls(product, localU[j * DIM], coefficient, DIM);
          PipeBarrier<PIPE_V>();
          Add(localU[row * DIM], localU[row * DIM], product, DIM);
          PipeBarrier<PIPE_V>();
          Muls(product, localW[j * DIM], coefficient, DIM);
          PipeBarrier<PIPE_V>();
          Add(localW[row * DIM], localW[row * DIM], product, DIM);
          PipeBarrier<PIPE_V>();
        }
        Cast(rowHalf, localU[row * DIM], RoundMode::CAST_NONE, DIM);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outputU[(headOffset + row) * DIM], rowHalf, DIM);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Cast(rowHalf, localW[row * DIM], RoundMode::CAST_NONE, DIM);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outputW[(headOffset + row) * DIM], rowHalf, DIM);
        // rowHalf is reused by MTE2 next, and localG/localGram by the next
        // task. Retain both dependencies; CPU stubs cannot validate events.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
      PipeBarrier<PIPE_ALL>();
    }
  }
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_fused_wy_v1(GM_ADDR k, GM_ADDR v, GM_ADDR gram, GM_ADDR g, GM_ADDR beta,
                                                       GM_ADDR w, GM_ADDR u, GM_ADDR config) {
  AscendC::InitSocState();
  FusedWY op;
  op.Run(k, v, gram, g, beta, w, u, config);
}
