// SPDX-License-Identifier: Apache-2.0
// Core-owned route tiles: contiguous stores and stable combine before reuse.
#define GLM_CHANNEL_FUSED_GATE_UP
#include "glm_channel_projection.cpp"
namespace {
constexpr uint32_t FLOAT_LANES = 64;
class FusedDownPrefill : public Projection {
 public:
  __aicore__ inline void RunFused(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR sums,
                                  GM_ADDR ends, GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets, GM_ADDR tiles,
                                  GM_ADDR ranks, GM_ADDR weights, GM_ADDR output, const __gm__ int64_t* config) {
    if (!Init(low, high, xs, codes, scales, sums, ends, order, sortedKeys, offsets, output, config)) return;
    if (topK_ > LANES || originalTokens_) return;
    tiles_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(tiles));
    ranks_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranks));
    routeWeights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    outputFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(combine_, 2 * N * sizeof(float));
    pipe_.InitBuffer(factors_, (LANES + LANES * LANES) * sizeof(float));
#ifdef GLM_CHANNEL_CACHE_PREFILL_ROUTES
    constexpr uint32_t CACHE_TOKENS = 640;
    pipe_.InitBuffer(rankCache_, CACHE_TOKENS * LANES * sizeof(int32_t));
    pipe_.InitBuffer(weightCache_, CACHE_TOKENS * LANES * sizeof(float));
    const bool cachedRoutes = tokens_ <= CACHE_TOKENS && topK_ == LANES;
    if (cachedRoutes) {
      DataCopy(rankCache_.Get<int32_t>(), ranks_, tokens_ * LANES);
      DataCopy(weightCache_.Get<float>(), routeWeights_, tokens_ * LANES);
      PipeBarrier<PIPE_ALL>();
    }
#endif
    const uint32_t maxRows = bits_ == 8 ? ROWS / 2 : ROWS;
    const int64_t valid = ends_.GetValue(experts_ - 1);
    const int64_t tileBase = GetBlockIdx() * tokens_ * topK_ * N;
    const BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
    for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
      int64_t first = 0;
      for (int64_t expert = 0; expert < experts_ && first < valid; ++expert) {
        const int64_t end = ends_.GetValue(expert);
        if (end < first || end > valid) return;
        for (int64_t row = first; row < end; row += maxRows) {
          const uint32_t count = end - row < maxRows ? end - row : maxRows;
          ProjectBlock(row, count, expert, tile);
          DataCopy(tiles_[tileBase + row * N], outputUb_.Get<half>(), count * N);
          PipeBarrier<PIPE_ALL>();
        }
        first = end;
      }
      // Only this core writes or reads its slot; no cross-core barrier or
      // atomic accumulation is needed, and it is reused only after combine.
      for (int64_t token = 0; token < tokens_; ++token) {
        auto value = combine_.Get<float>();
        auto accum = value[N];
        Duplicate(accum, 0.0f, N);
#ifdef GLM_CHANNEL_CACHE_PREFILL_ROUTES
        if (cachedRoutes) {
          Brcb(factors_.Get<float>()[LANES], weightCache_.Get<float>()[token * LANES], 1, {1, LANES});
        } else {
#endif
          DataCopy(factors_.Get<float>(), routeWeights_[token * LANES], LANES);
          PipeBarrier<PIPE_ALL>();
          Brcb(factors_.Get<float>()[LANES], factors_.Get<float>(), 1, {1, LANES});
#ifdef GLM_CHANNEL_CACHE_PREFILL_ROUTES
        }
#endif
        PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < topK_; ++slot) {
#ifdef GLM_CHANNEL_CACHE_PREFILL_ROUTES
          const int64_t row = cachedRoutes ? rankCache_.Get<int32_t>().GetValue(token * LANES + slot)
                                           : ranks_.GetValue(token * topK_ + slot);
#else
          const int64_t row = ranks_.GetValue(token * topK_ + slot);
#endif
          if (row < 0 || row >= valid) continue;
          DataCopy(outputUb_.Get<half>(), tiles_[tileBase + row * N], N);
          PipeBarrier<PIPE_ALL>();
          Cast(value, outputUb_.Get<half>(), RoundMode::CAST_NONE, N);
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
  TBuf<TPosition::VECCALC> combine_, factors_;
#ifdef GLM_CHANNEL_CACHE_PREFILL_ROUTES
  TBuf<TPosition::VECCALC> rankCache_, weightCache_;
#endif
  GlobalTensor<half> tiles_;
  GlobalTensor<int32_t> ranks_;
  GlobalTensor<float> routeWeights_, outputFloat_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_channel_down_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                                  GM_ADDR scales, GM_ADDR sums, GM_ADDR ends,
                                                                  GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets,
                                                                  GM_ADDR tiles, GM_ADDR ranks, GM_ADDR weights,
                                                                  GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  FusedDownPrefill operation;
  operation.RunFused(low, high, xs, codes, scales, sums, ends, order, sortedKeys, offsets, tiles, ranks, weights,
                     output, reinterpret_cast<__gm__ int64_t*>(config));
}
