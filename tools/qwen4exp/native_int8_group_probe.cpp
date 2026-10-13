// SPDX-License-Identifier: Apache-2.0
// Arithmetic probe only: prepared INT8 G128 operands, exact INT32 products.
// No quantization, routing, scaling, full-K reduction, or serving substitution.
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t M = 16, N = 64, K = 128, BLOCK = 16, K0 = 32;
class Int8GroupProbe {
 public:
  __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR output) {
    GlobalTensor<int8_t> input, weight;
    GlobalTensor<int32_t> destination;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a));
    weight.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b));
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
    pipe_.InitBuffer(a1_, M * K);
    pipe_.InitBuffer(b1_, N * K);
    pipe_.InitBuffer(a2_, M * K);
    pipe_.InitBuffer(b2_, N * K);
    pipe_.InitBuffer(c_, M * N * sizeof(int32_t));
    pipe_.InitBuffer(result_, M * N * sizeof(int32_t));
    DataCopy(a1_.Get<int8_t>(), input, M * K);
    DataCopy(b1_.Get<int8_t>(), weight, N * K);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData2DParams loadA;
    loadA.repeatTimes = K / K0;
    loadA.srcStride = M / BLOCK;
    loadA.ifTranspose = false;
    LoadData(a2_.Get<int8_t>(), a1_.Get<int8_t>(), loadA);
    LoadData2DParams loadB;
    loadB.repeatTimes = N / BLOCK;
    loadB.srcStride = K / K0;
    loadB.ifTranspose = false;
    for (uint32_t kb = 0; kb < K / K0; ++kb) {
      LoadData(b2_.Get<int8_t>()[kb * N * K0], b1_.Get<int8_t>()[kb * BLOCK * K0], loadB);
    }
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    MmadParams mm;
    mm.m = M;
    mm.n = N;
    mm.k = K;
    mm.cmatrixInitVal = true;
    Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>(), b2_.Get<int8_t>(), mm);
    SetFlag<HardEvent::M_V>(EVENT_ID0);
    WaitFlag<HardEvent::M_V>(EVENT_ID0);
    DataCopyParams copy{N / BLOCK, M / BLOCK, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(result_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(destination, result_.Get<int32_t>(), M * N);
    pipe_.Destroy();
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> result_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_int8_group_probe_v1(GM_ADDR a, GM_ADDR b, GM_ADDR output) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8GroupProbe operation;
  operation.Run(a, b, output);
}
