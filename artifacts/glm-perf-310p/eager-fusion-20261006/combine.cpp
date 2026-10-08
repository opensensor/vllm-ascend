// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
// SPDX-License-Identifier: Apache-2.0
#ifndef W2_ROUTE_COMBINE_GEOMETRY_H
#define W2_ROUTE_COMBINE_GEOMETRY_H
#include <cstdint>
namespace NsW2Combine {
constexpr int64_t MAX_ROUTES = 32768;
constexpr int64_t MAX_WIDTH = 16384;
constexpr int64_t MAX_TOP_K = 32;
constexpr int64_t MAX_EXPERTS = 1024;
constexpr int64_t CHANNEL_ALIGNMENT = 32;
constexpr int64_t CHANNEL_TILE = 1024;
constexpr bool ValidGeometry(int64_t rows, int64_t hidden, int64_t tokens, int64_t topK, int64_t experts) {
    return rows > 0 && rows <= MAX_ROUTES && hidden > 0 && hidden <= MAX_WIDTH &&
           hidden % CHANNEL_ALIGNMENT == 0 && tokens > 0 && tokens <= MAX_ROUTES &&
           topK > 0 && topK <= MAX_TOP_K && rows == tokens * topK &&
           experts > 0 && experts <= MAX_EXPERTS;
}
}  // namespace NsW2Combine
#endif

namespace {
using namespace AscendC;
constexpr uint32_t TILE = NsW2Combine::CHANNEL_TILE;
constexpr uint32_t STORAGE_BYTES = TILE * (sizeof(half) + 2 * sizeof(float));
struct RouteCombineTiling { int64_t tokens, hidden, top_k, experts; };

class RouteCombine {
 public:
  __aicore__ inline void Run(GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights, GM_ADDR ends, GM_ADDR y,
                            const RouteCombineTiling& td) {
    routed_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(routed));
    inverse_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(inverse));
    weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    pipe_.InitBuffer(storage_, STORAGE_BYTES);
    auto input = storage_.Get<half>();
    auto product = storage_.Get<float>()[TILE / 2];
    auto sum = storage_.Get<float>()[TILE / 2 + TILE];
    const int64_t validRows = ends_.GetValue(td.experts - 1);
    ASCENDC_ASSERT(validRows >= 0 && validRows <= td.tokens * td.top_k,
                   { KERNEL_LOG(KERNEL_ERROR, "invalid local route boundary"); });
    const int64_t tiles = (td.hidden + TILE - 1) / TILE;
    for (int64_t task = GetBlockIdx(); task < td.tokens * tiles; task += GetBlockNum()) {
      const int64_t token = task / tiles;
      const int64_t column = (task % tiles) * TILE;
      const uint32_t count = td.hidden - column < TILE ? td.hidden - column : TILE;
      Duplicate(sum, 0.0f, count);
      PipeBarrier<PIPE_V>();
      for (int64_t slot = 0; slot < td.top_k; ++slot) {
        const int64_t route = token * td.top_k + slot;
        const int64_t row = inverse_.GetValue(route);
        ASCENDC_ASSERT(row >= 0 && row < td.tokens * td.top_k,
                       { KERNEL_LOG(KERNEL_ERROR, "invalid inverse route index"); });
        const float weight = weights_.GetValue(route);
        // Peer rows can be uninitialized: inspect metadata before reading them.
        if (row >= validRows || weight == 0.0f) continue;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(input, routed_[row * td.hidden + column], count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(product, input, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(product, product, weight, count);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, product, count);
        PipeBarrier<PIPE_V>();
      }
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(y_[token * td.hidden + column], sum, count);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> routed_;
  GlobalTensor<int64_t> inverse_, ends_;
  GlobalTensor<float> weights_, y_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_resident_combine_v1(GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights, GM_ADDR ends, GM_ADDR y, GM_ADDR config) {
  AscendC::InitSocState();
  auto raw = reinterpret_cast<__gm__ RouteCombineTiling*>(config);
  const RouteCombineTiling td{raw->tokens, raw->hidden, raw->top_k, raw->experts};
  RouteCombine op;
  op.Run(routed, inverse, weights, ends, y, td);
}
