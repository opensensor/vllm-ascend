// SPDX-License-Identifier: Apache-2.0
// Decode only: down projection and stable original-slot combine stay in UB.
#define GLM_CHANNEL_FUSED_GATE_UP
#include "glm_channel_projection.cpp"
namespace {
constexpr uint32_t DECODE_TOKENS = 16, TOP_K = 8, FLOAT_LANES = 64;
class FusedDownDecode : public Projection {
 public:
  __aicore__ inline void RunFused(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR sums,
                                  GM_ADDR ends, GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets, GM_ADDR weights,
                                  GM_ADDR output, const __gm__ int64_t* config) {
    if (!Init(low, high, xs, codes, scales, sums, ends, order, sortedKeys, offsets, output, config)) return;
    if (tokens_ > DECODE_TOKENS || topK_ > TOP_K || originalTokens_) return;
    outputFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    routeWeights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    pipe_.InitBuffer(slots_, DECODE_TOKENS * TOP_K * N * sizeof(half));
    pipe_.InitBuffer(combine_, 2 * N * sizeof(float));
    pipe_.InitBuffer(factors_, (LANES + LANES * LANES) * sizeof(float));
    const uint32_t maxRows = bits_ == 8 ? ROWS / 2 : ROWS;
    const int64_t valid = ends_.GetValue(experts_ - 1);
    const BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
    for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
      Duplicate(slots_.Get<half>(), static_cast<half>(0), tokens_ * topK_ * N);
      PipeBarrier<PIPE_V>();
      int64_t first = 0;
      while (first < valid) {
        const int64_t expert = static_cast<int64_t>(sortedKeys_.GetValue(first));
        if (expert < 0 || expert >= experts_) return;
        const int64_t end = ends_.GetValue(expert);
        if (end <= first || end > valid) return;
        for (int64_t row = first; row < end; row += maxRows) {
          const uint32_t count = end - row < maxRows ? end - row : maxRows;
          ProjectBlock(row, count, expert, tile);
          for (uint32_t m = 0; m < count; ++m) {
            const int64_t position = order_.GetValue(row + m);
            if (position < 0 || position >= tokens_ * topK_) return;
            Adds(slots_.Get<half>()[position * N], outputUb_.Get<half>()[m * N], static_cast<half>(0), N);
          }
          PipeBarrier<PIPE_V>();
        }
        first = end;
      }
      for (int64_t token = 0; token < tokens_; ++token) {
        auto value = combine_.Get<float>();
        auto accum = value[N];
        Duplicate(accum, 0.0f, N);
        DataCopy(factors_.Get<float>(), routeWeights_[token * LANES], LANES);
        PipeBarrier<PIPE_ALL>();
        Brcb(factors_.Get<float>()[LANES], factors_.Get<float>(), 1, {1, LANES});
        PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < topK_; ++slot) {
          Cast(value, slots_.Get<half>()[(token * topK_ + slot) * N], RoundMode::CAST_NONE, N);
          PipeBarrier<PIPE_V>();
          Mul(value, value, factors_.Get<float>()[LANES + slot * LANES], static_cast<uint64_t>(FLOAT_LANES),
              N / FLOAT_LANES, broadcast);
          PipeBarrier<PIPE_V>();
          Add(accum, accum, value, N);
          PipeBarrier<PIPE_V>();
        }
        DataCopy(outputFloat_[token * n_ + tile * N], accum, N);
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

 private:
  TBuf<TPosition::VECCALC> slots_, combine_, factors_;
  GlobalTensor<float> outputFloat_, routeWeights_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_channel_down_decode_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                                 GM_ADDR scales, GM_ADDR sums, GM_ADDR ends,
                                                                 GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets,
                                                                 GM_ADDR weights, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  FusedDownDecode operation;
  operation.RunFused(low, high, xs, codes, scales, sums, ends, order, sortedKeys, offsets, weights, output,
                     reinterpret_cast<__gm__ int64_t*>(config));
}
