// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t HIDDEN_TILE = 512;

class QwenW8RouteCombine {
 public:
  __aicore__ inline void Run(GM_ADDR sortedRoutes, GM_ADDR inverseOrder, GM_ADDR routeWeights,
                             GM_ADDR localRouteMask, GM_ADDR output, int64_t numTokens,
                             int64_t topK, int64_t hiddenSize) {
    sortedRoutes_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sortedRoutes));
    inverseOrder_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(inverseOrder));
    routeWeights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routeWeights));
    localRouteMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(localRouteMask));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    pipe_.InitBuffer(routeBuffer_, HIDDEN_TILE * sizeof(half));
    pipe_.InitBuffer(valueBuffer_, HIDDEN_TILE * sizeof(float));
    pipe_.InitBuffer(accumulatorBuffer_, HIDDEN_TILE * sizeof(float));

    const int64_t tilesPerToken = (hiddenSize + HIDDEN_TILE - 1) / HIDDEN_TILE;
    const int64_t taskCount = numTokens * tilesPerToken;
    auto route = routeBuffer_.Get<half>();
    auto values = valueBuffer_.Get<float>();
    auto accumulator = accumulatorBuffer_.Get<float>();
    for (int64_t task = GetBlockIdx(); task < taskCount; task += GetBlockNum()) {
      const int64_t token = task / tilesPerToken;
      const int64_t tile = task - token * tilesPerToken;
      const int64_t hiddenOffset = tile * HIDDEN_TILE;
      const uint32_t count = static_cast<uint32_t>(
          hiddenSize - hiddenOffset < HIDDEN_TILE ? hiddenSize - hiddenOffset : HIDDEN_TILE);
      Duplicate(accumulator, 0.0f, count);
      PipeBarrier<PIPE_V>();

      // Preserve original top-k order. Peer rows can contain undefined bytes
      // after grouped matmul, so test the localized route id before reading.
      for (int64_t k = 0; k < topK; ++k) {
        const int64_t slot = token * topK + k;
        if (localRouteMask_.GetValue(slot) == 0) {
          continue;
        }
        const int32_t sortedRow = inverseOrder_.GetValue(slot);
        if (sortedRow < 0 || sortedRow >= numTokens * topK) {
          continue;
        }
        DataCopy(route, sortedRoutes_[static_cast<int64_t>(sortedRow) * hiddenSize + hiddenOffset], count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(values, route, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        // Existing eager code rounds each FP32 router weight through FP16
        // before multiplying FP32 route payloads. Keep that rounding point.
        const half roundedWeight = static_cast<half>(routeWeights_.GetValue(slot));
        Muls(values, values, static_cast<float>(roundedWeight), count);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, values, count);
        PipeBarrier<PIPE_V>();
        // The next route reuses ``route`` through MTE2 while the vector pipe
        // may still be consuming it. A V->MTE2 dependency is required on
        // dav-m200; command issue order alone does not protect the shared UB
        // buffer and otherwise makes multi-route sums nondeterministic.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
      }
      Cast(route, accumulator, RoundMode::CAST_NONE, count);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output_[token * hiddenSize + hiddenOffset], route, count);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      // ``route`` is both the MTE3 source above and the MTE2 destination for
      // the next task. MTE3_V only protects the next vector Cast; it does not
      // stop a following MTE2 DataCopy from overwriting this tile before MTE3
      // has finished reading it. This cross-task dependency is observable as
      // corrupted early tiles while the last tile handled by each core is
      // correct.
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> routeBuffer_, valueBuffer_, accumulatorBuffer_;
  GlobalTensor<half> sortedRoutes_, output_;
  GlobalTensor<int32_t> inverseOrder_;
  GlobalTensor<uint8_t> localRouteMask_;
  GlobalTensor<float> routeWeights_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_w8_route_combine_v310(
    GM_ADDR sortedRoutes, GM_ADDR inverseOrder, GM_ADDR routeWeights, GM_ADDR localRouteMask,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  auto td = reinterpret_cast<__gm__ int64_t*>(tiling);
  QwenW8RouteCombine op;
  op.Run(sortedRoutes, inverseOrder, routeWeights, localRouteMask, output, td[0], td[1], td[2]);
}
