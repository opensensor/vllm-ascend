// SPDX-License-Identifier: Apache-2.0
// Standalone comparison of the deployed scalar beta multiply and a vector row.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int32_t MAX_CHANNELS = 256;
class KdaBetaScale {
 public:
  __aicore__ inline void Run(GM_ADDR source, GM_ADDR beta, GM_ADDR destination, GM_ADDR descriptor, bool scalar) {
    auto config = reinterpret_cast<__gm__ int64_t*>(descriptor);
    const int64_t rows = config[0], channels = config[1];
    const int64_t sourceStride = config[2], destinationStride = config[3], betaStride = config[4];
    if (rows <= 0 || channels < 16 || channels > MAX_CHANNELS || channels % 16) return;
    GlobalTensor<half> input, output;
    GlobalTensor<float> scales;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(source));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(destination));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
    pipe_.InitBuffer(inputBuffer_, MAX_CHANNELS * sizeof(half));
    pipe_.InitBuffer(floatBuffer_, MAX_CHANNELS * sizeof(float));
    pipe_.InitBuffer(outputBuffer_, MAX_CHANNELS * sizeof(half));
    auto inputLocal = inputBuffer_.Get<half>();
    auto floatLocal = floatBuffer_.Get<float>();
    auto outputLocal = outputBuffer_.Get<half>();
    for (int64_t row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
      const float betaScale = scales.GetValue(row * betaStride);
      if (scalar) {
        // Same arithmetic as ScaleTailRowsByBeta310P, including the terminal
        // static_cast used by FloatToType<half>. No beta pre-rounding.
        for (int64_t column = 0; column < channels; ++column) {
          const float value = static_cast<float>(input.GetValue(row * sourceStride + column));
          output.SetValue(row * destinationStride + column, static_cast<half>(value * betaScale));
        }
      } else {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        DataCopy(inputLocal, input[row * sourceStride], static_cast<uint32_t>(channels));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(floatLocal, inputLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(channels));
        PipeBarrier<PIPE_V>();
        Muls(floatLocal, floatLocal, betaScale, static_cast<uint32_t>(channels));
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, floatLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(channels));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(output[row * destinationStride], outputLocal, static_cast<uint32_t>(channels));
      }
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputBuffer_, floatBuffer_, outputBuffer_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_kda_beta_scale_v1(GM_ADDR source, GM_ADDR beta, GM_ADDR output,
                                                            GM_ADDR config) {
  AscendC::InitSocState();
  KdaBetaScale operation;
#ifdef GLM_KDA_BETA_SCALAR_REFERENCE
  operation.Run(source, beta, output, config, true);
#else
  operation.Run(source, beta, output, config, false);
#endif
}
