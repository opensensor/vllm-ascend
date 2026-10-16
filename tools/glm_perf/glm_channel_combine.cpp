// SPDX-License-Identifier: Apache-2.0
// Stable original top-k order; all weighting and accumulation stays vector FP32.
#include "kernel_operator.h"
#include "glm_channel_dma.h"
namespace {
using namespace AscendC;
constexpr uint32_t N = 128, LANES = 8, FLOAT_LANES = 64;
class Combine {
 public:
  __aicore__ inline void Run(GM_ADDR routed, GM_ADDR ranks, GM_ADDR ends, GM_ADDR weights, GM_ADDR output,
                             const __gm__ int64_t* config) {
    const int64_t tokens = config[0], topK = config[1], experts = config[2], hidden = config[3];
    if (tokens < 1 || topK < 1 || topK > LANES || experts < 1 || hidden % N) return;
    routed_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(routed));
    ranks_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranks));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ends));
    weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(half_, N * sizeof(half));
    pipe_.InitBuffer(value_, N * sizeof(float));
    pipe_.InitBuffer(accum_, N * sizeof(float));
    pipe_.InitBuffer(scale_, (LANES + LANES * LANES) * sizeof(float));
    const int64_t valid = ends_.GetValue(experts - 1);
    const BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
    for (int64_t task = GetBlockIdx(); task < tokens * (hidden / N); task += GetBlockNum()) {
      const int64_t token = task / (hidden / N), column = task % (hidden / N) * N;
      Duplicate(accum_.Get<float>(), 0.0f, N);
      DataCopy(scale_.Get<float>(), weights_[token * LANES], LANES);
      GlmChannelDma::Fence();
      Brcb(scale_.Get<float>()[LANES], scale_.Get<float>(), 1, {1, LANES});
      PipeBarrier<PIPE_V>();
      for (int64_t slot = 0; slot < topK; ++slot) {
        const int64_t row = ranks_.GetValue(token * topK + slot);
        if (row < 0 || row >= valid) continue;
        GlmChannelDma::BeforeRead();
        DataCopy(half_.Get<half>(), routed_[row * hidden + column], N);
        GlmChannelDma::Fence();
        Cast(value_.Get<float>(), half_.Get<half>(), RoundMode::CAST_NONE, N);
        PipeBarrier<PIPE_V>();
        Mul(value_.Get<float>(), value_.Get<float>(), scale_.Get<float>()[LANES + slot * LANES],
            static_cast<uint64_t>(FLOAT_LANES), N / FLOAT_LANES, broadcast);
        PipeBarrier<PIPE_V>();
        Add(accum_.Get<float>(), accum_.Get<float>(), value_.Get<float>(), N);
        PipeBarrier<PIPE_V>();
      }
      GlmChannelDma::WriteReady();
      DataCopy(output_[token * hidden + column], accum_.Get<float>(), N);
      GlmChannelDma::Fence();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> half_, value_, accum_, scale_;
  GlobalTensor<half> routed_;
  GlobalTensor<int32_t> ranks_;
  GlobalTensor<int32_t> ends_;
  GlobalTensor<float> weights_, output_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_channel_combine_v1(GM_ADDR routed, GM_ADDR ranks, GM_ADDR ends,
                                                             GM_ADDR weights, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  Combine operation;
  operation.Run(routed, ranks, ends, weights, output, reinterpret_cast<__gm__ int64_t*>(config));
}
