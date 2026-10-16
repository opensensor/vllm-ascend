// SPDX-License-Identifier: Apache-2.0
// Token windows retain all original route slots in UB; no GM route output.
#define GLM_CHANNEL_FUSED_GATE_UP
#define GLM_CHANNEL_TOKEN_PREFILL
#include "glm_channel_projection.cpp"
#include "glm_channel_dma.h"
namespace {
constexpr uint32_t TOKEN_WINDOW = 16, WINDOW_ROUTES = TOKEN_WINDOW * LANES, FLOAT_LANES = 64;
class TokenDown : public Projection {
 public:
  __aicore__ inline void RunToken(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR sums,
                                  GM_ADDR ends, GM_ADDR rows, GM_ADDR keys, GM_ADDR offsets, GM_ADDR unusedTiles,
                                  GM_ADDR positions, GM_ADDR weights, GM_ADDR output, const __gm__ int64_t* config) {
    if (!Init(low, high, xs, codes, scales, sums, ends, rows, keys, offsets, output, config)) return;
    if (topK_ != LANES || originalTokens_) return;
    GlobalTensor<int32_t> windowRows;
    GlobalTensor<int64_t> windowPositions;
    GlobalTensor<float> routeWeights, result;
    windowRows.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rows));
    windowPositions.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
    routeWeights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(tokenRowsUb_, WINDOW_ROUTES * sizeof(int32_t));
    pipe_.InitBuffer(positions_, WINDOW_ROUTES * sizeof(int64_t));
    pipe_.InitBuffer(slots_, WINDOW_ROUTES * N * sizeof(half));
    pipe_.InitBuffer(validSlots_, WINDOW_ROUTES * sizeof(int32_t));
    pipe_.InitBuffer(combine_, 2 * N * sizeof(float));
    pipe_.InitBuffer(factors_, (WINDOW_ROUTES + LANES * LANES) * sizeof(float));
    const uint32_t maxRows = bits_ == 8 ? ROWS / 2 : ROWS;
    const int64_t valid = ends_.GetValue(experts_ - 1);
    const BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
    for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
      for (int64_t firstToken = 0; firstToken < tokens_; firstToken += TOKEN_WINDOW) {
        const uint32_t tokenCount = tokens_ - firstToken < TOKEN_WINDOW ? tokens_ - firstToken : TOKEN_WINDOW;
        const int64_t window = firstToken / TOKEN_WINDOW;
        // CPU-free frontend pads and sorts this fixed window by global route
        // rank. Every expert's routes are contiguous, including repeated slots.
        DataCopy(tokenRowsUb_.Get<int32_t>(), windowRows[window * WINDOW_ROUTES], WINDOW_ROUTES);
        DataCopy(positions_.Get<int64_t>(), windowPositions[window * WINDOW_ROUTES], WINDOW_ROUTES);
        DataCopy(factors_.Get<float>(), routeWeights[firstToken * LANES], tokenCount * LANES);
        Duplicate(slots_.Get<half>(), static_cast<half>(0), tokenCount * LANES * N);
        Duplicate(validSlots_.Get<int32_t>(), static_cast<int32_t>(0), WINDOW_ROUTES);
        GlmChannelDma::Fence();
        uint32_t cursor = 0;
        while (cursor < tokenCount * LANES) {
          const int64_t globalRow = tokenRowsUb_.Get<int32_t>().GetValue(cursor);
          if (globalRow < 0 || globalRow >= valid) break;
          const int64_t expert = sortedKeys_.GetValue(globalRow);
          if (expert < 0 || expert >= experts_) return;
          const int64_t expertEnd = ends_.GetValue(expert);
          uint32_t stop = cursor;
          while (stop < tokenCount * LANES && tokenRowsUb_.Get<int32_t>().GetValue(stop) < expertEnd) ++stop;
          if (stop == cursor) return;
          for (uint32_t chunk = cursor; chunk < stop; chunk += maxRows) {
            const uint32_t count = stop - chunk < maxRows ? stop - chunk : maxRows;
            indirectRows_ = true;
            indirectStart_ = chunk;
            ProjectBlock(0, count, expert, tile);
            for (uint32_t m = 0; m < count; ++m) {
              const int64_t position = positions_.Get<int64_t>().GetValue(chunk + m);
              if (position < 0 || position >= tokenCount * LANES) return;
              validSlots_.Get<int32_t>().SetValue(position, 1);
              Adds(slots_.Get<half>()[position * N], outputUb_.Get<half>()[m * N], static_cast<half>(0), N);
            }
            PipeBarrier<PIPE_V>();
          }
          cursor = stop;
        }
        for (uint32_t token = 0; token < tokenCount; ++token) {
          auto value = combine_.Get<float>();
          auto accum = value[N];
          Duplicate(accum, 0.0f, N);
          Brcb(factors_.Get<float>()[WINDOW_ROUTES], factors_.Get<float>()[token * LANES], 1, {1, LANES});
          PipeBarrier<PIPE_V>();
          for (uint32_t slot = 0; slot < LANES; ++slot) {
            if (!validSlots_.Get<int32_t>().GetValue(token * LANES + slot)) continue;
            Cast(value, slots_.Get<half>()[(token * LANES + slot) * N], RoundMode::CAST_NONE, N);
            PipeBarrier<PIPE_V>();
            Mul(value, value, factors_.Get<float>()[WINDOW_ROUTES + slot * LANES], static_cast<uint64_t>(FLOAT_LANES),
                N / FLOAT_LANES, broadcast);
            PipeBarrier<PIPE_V>();
            Add(accum, accum, value, N);
            PipeBarrier<PIPE_V>();
          }
          GlmChannelDma::WriteReady();
          DataCopy(result[(firstToken + token) * n_ + tile * N], accum, N);
          GlmChannelDma::Fence();
        }
      }
    }
  }

 private:
  TBuf<TPosition::VECCALC> positions_, slots_, combine_, factors_, validSlots_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_channel_down_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                                  GM_ADDR scales, GM_ADDR sums, GM_ADDR ends,
                                                                  GM_ADDR rows, GM_ADDR keys, GM_ADDR offsets,
                                                                  GM_ADDR unusedTiles, GM_ADDR positions,
                                                                  GM_ADDR weights, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  TokenDown operation;
  operation.RunToken(low, high, xs, codes, scales, sums, ends, rows, keys, offsets, unusedTiles, positions, weights,
                     output, reinterpret_cast<__gm__ int64_t*>(config));
}
