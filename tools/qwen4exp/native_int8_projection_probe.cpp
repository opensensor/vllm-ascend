// SPDX-License-Identifier: Apache-2.0
// Experimental single-expert G128 projection; prepared INT8 operands only.
// All groups accumulate in FP32, with the same sw-then-sx multiplication order.
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t M = 16, N = 160, GROUP = 128, BLOCK = 16, K0 = 32, LANES = 8, MAX_K = 2560;
constexpr uint32_t ELEMENTS = M * N;
class Int8ProjectionProbe {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR inputScale, GM_ADDR output,
                             GM_ADDR descriptor) {
    auto config = reinterpret_cast<__gm__ int64_t*>(descriptor);
    const int64_t rows = config[0], k = config[1], n = config[2], groups = k / GROUP;
    ASCENDC_ASSERT(rows > 0 && k > 0 && k <= MAX_K && k % GROUP == 0 && n > 0 && n % N == 0,
                   { KERNEL_LOG(KERNEL_ERROR, "invalid INT8 projection probe geometry"); });
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(weight));
    sw_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
    sx_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputScale));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    pipe_.InitBuffer(a1_, M * GROUP);
    pipe_.InitBuffer(b1_, N * MAX_K);
    pipe_.InitBuffer(a2_, M * GROUP);
    pipe_.InitBuffer(b2_, N * GROUP);
    pipe_.InitBuffer(c_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(packed_, M * GROUP);
    pipe_.InitBuffer(integer_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(product_, 2 * ELEMENTS * sizeof(float));
    pipe_.InitBuffer(metadata_, N * sizeof(half) + N * sizeof(float) + M * LANES * sizeof(float));
    pipe_.InitBuffer(result_, ELEMENTS * sizeof(half));
    auto packed = packed_.Get<int8_t>();
    auto product = product_.Get<float>();
    auto accumulator = product[ELEMENTS];
    auto scales = metadata_.Get<half>();
    auto weightScale = metadata_.Get<float>()[N / 2];
    auto activationScale = weightScale[N];
    for (int64_t tile = GetBlockIdx(); tile < n / N; tile += GetBlockNum()) {
      // One resident full-K tile; no per-group reads of the weight bank from GM.
      DataCopy(b1_.Get<int8_t>(), w_[tile * N * k], N * k);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      for (int64_t first = 0; first < rows; first += M) {
        const uint32_t count = rows - first < M ? rows - first : M;
        Duplicate(accumulator, 0.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (int64_t group = 0; group < groups; ++group) {
          Duplicate(packed.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), M * GROUP / 2);
          Duplicate(activationScale, 0.0f, M * LANES);
          SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
          DataCopyParams inputCopy{static_cast<uint16_t>(count), 1, static_cast<uint16_t>(k / K0 - 1), 0};
          for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
            DataCopy(packed[kb * M * K0], x_[first * k + group * GROUP + kb * K0], inputCopy);
          }
          DataCopyParams scaleCopy{N / BLOCK, 1, static_cast<uint16_t>(groups - 1), 0};
          DataCopy(scales, sw_[(tile * N / BLOCK * groups + group) * BLOCK], scaleCopy);
          DataCopyParams activationCopy{static_cast<uint16_t>(count), 1, static_cast<uint16_t>(groups - 1), 0};
          DataCopy(activationScale, sx_[(first * groups + group) * LANES], activationCopy);
          SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
          DataCopy(a1_.Get<int8_t>(), packed, M * GROUP);
          SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
          LoadData2DParams loadA;
          loadA.repeatTimes = GROUP / K0;
          loadA.srcStride = M / BLOCK;
          loadA.ifTranspose = false;
          LoadData(a2_.Get<int8_t>(), a1_.Get<int8_t>(), loadA);
          LoadData2DParams loadB;
          loadB.repeatTimes = N / BLOCK;
          loadB.srcStride = k / K0;
          loadB.ifTranspose = false;
          for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
            LoadData(b2_.Get<int8_t>()[kb * N * K0], b1_.Get<int8_t>()[(group * GROUP + kb * K0) * BLOCK], loadB);
          }
          SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
          WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
          MmadParams mm;
          mm.m = M;
          mm.n = N;
          mm.k = GROUP;
          mm.cmatrixInitVal = true;
          Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>(), b2_.Get<int8_t>(), mm);
          SetFlag<HardEvent::M_V>(EVENT_ID0);
          WaitFlag<HardEvent::M_V>(EVENT_ID0);
          DataCopyParams copy{N / BLOCK, M / BLOCK, 0, 0};
          DataCopyEnhancedParams enhanced;
          enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
          DataCopy(integer_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
          PipeBarrier<PIPE_V>();
          Cast(product, integer_.Get<int32_t>(), RoundMode::CAST_NONE, ELEMENTS);
          Cast(weightScale, scales, RoundMode::CAST_NONE, N);
          PipeBarrier<PIPE_V>();
          for (uint32_t row = 0; row < count; ++row)
            Mul(product[row * BLOCK], product[row * BLOCK], weightScale, BLOCK, N / BLOCK,
                {1, 1, 1, M * BLOCK / LANES, M * BLOCK / LANES, BLOCK / LANES});
          PipeBarrier<PIPE_V>();
          for (uint32_t row = 0; row < count; ++row)
            Mul(product[row * BLOCK], product[row * BLOCK], activationScale[row * LANES], BLOCK, N / BLOCK,
                {1, 1, 0, M * BLOCK / LANES, M * BLOCK / LANES, 0});
          PipeBarrier<PIPE_V>();
          Add(accumulator, accumulator, product, ELEMENTS);
          PipeBarrier<PIPE_V>();
          SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
          WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
          SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Cast(result_.Get<half>(), accumulator, RoundMode::CAST_NONE, ELEMENTS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams store{static_cast<uint16_t>(count), 1, 0, static_cast<uint16_t>(n / BLOCK - 1)};
        for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
          DataCopy(y_[first * n + tile * N + nb * BLOCK], result_.Get<half>()[nb * M * BLOCK], store);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      }
      SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
    pipe_.Destroy();
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> packed_, integer_, product_, metadata_, result_;
  GlobalTensor<int8_t> x_, w_;
  GlobalTensor<half> sw_, y_;
  GlobalTensor<float> sx_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_int8_projection_probe_v1(GM_ADDR x, GM_ADDR w, GM_ADDR sw, GM_ADDR sx,
                                                                    GM_ADDR output, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe operation;
  operation.Run(x, w, sw, sx, output, descriptor);
}
