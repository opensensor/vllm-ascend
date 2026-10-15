// SPDX-License-Identifier: Apache-2.0
// FP16 -> FP32, route weight, inverse permutation. Reduction stays in PyTorch.
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t WIDTH = 2560;
constexpr int64_t MAX_ROUTES = 25600;

class Finalize {
 public:
  __aicore__ inline void Run(GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights, GM_ADDR output, int64_t routes) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(routed));
    inverse_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(inverse));
    weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(inputBuffer_, WIDTH * sizeof(half));
    pipe_.InitBuffer(outputBuffer_, WIDTH * sizeof(float));
    auto input = inputBuffer_.Get<half>();
    auto result = outputBuffer_.Get<float>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for (int64_t row = GetBlockIdx(); row < routes; row += GetBlockNum()) {
      const int64_t source = inverse_.GetValue(row);
      if (source < 0 || source >= routes) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid inverse route index"); });
        break;
      }
      const float weight = weights_.GetValue(row);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(input, input_[source * WIDTH], WIDTH);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      Cast(result, input, RoundMode::CAST_NONE, WIDTH);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      PipeBarrier<PIPE_V>();
      // One separate FP32 multiply: no reduction, half rounding, or FMA.
      Muls(result, result, weight, WIDTH);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output_[row * WIDTH], result, WIDTH);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputBuffer_, outputBuffer_;
  GlobalTensor<half> input_;
  GlobalTensor<float> output_, weights_;
  GlobalTensor<int64_t> inverse_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_route_finalize_v1(GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights,
                                                             GM_ADDR output, GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  const int64_t routes = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (routes <= 0 || routes > MAX_ROUTES) return;
  Finalize op;
  op.Run(routed, inverse, weights, output, routes);
}
