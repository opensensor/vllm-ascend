// SPDX-License-Identifier: Apache-2.0
// Offline candidate. Local-row finalization retains the external sum; fused
// entries use explicit FP32 trees that still require comparison with CANN.
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t WIDTH = 2560, TOP_K = 10, EXPERTS = 128;
constexpr int64_t MAX_ROUTES = 20480;
// ORDER=0 writes weighted routes. ORDER=1 folds from +0 in route order.
// ORDER=2 uses an adjacent-pair tree with odd nodes carried to the next level.
template <int ORDER>
class LocalEpilogue {
  // The unfused variant keeps the qualified finalizer's whole-row transfers.
  // Fused reductions tile columns to bound the ten weighted terms in UB.
  static constexpr uint32_t TILE = ORDER == 0 ? WIDTH : 256, COLUMNS = WIDTH / TILE;

 public:
  __aicore__ inline void Run(GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights, GM_ADDR ends, GM_ADDR output,
                             int64_t rows) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(routed));
    inverse_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(inverse));
    weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    const int64_t active = ends_.GetValue(EXPERTS - 1);
    if (active < 0 || active > rows) {
      ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid local epilogue boundary"); });
      return;
    }
    constexpr uint32_t TERMS = ORDER == 0 ? 1 : TOP_K;
    pipe_.InitBuffer(inputBuffer_, TILE * sizeof(half));
    pipe_.InitBuffer(termsBuffer_, TERMS * TILE * sizeof(float));
    pipe_.InitBuffer(sumBuffer_, TILE * sizeof(float));
    auto input = inputBuffer_.template Get<half>();
    auto terms = termsBuffer_.template Get<float>();
    auto sum = sumBuffer_.template Get<float>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    const int64_t tasks = (ORDER == 0 ? rows : rows / TOP_K) * COLUMNS;
    for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
      const int64_t outputRow = task / COLUMNS;
      const uint32_t column = task % COLUMNS * TILE;
      // The output of the previous task may alias terms or sum. Drain that
      // store before reusing either buffer, including the all-peer case.
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      for (uint32_t term = 0; term < TERMS; ++term) {
        const int64_t route = ORDER == 0 ? outputRow : outputRow * TOP_K + term;
        const int64_t source = inverse_.GetValue(route);
        if (source < 0 || source >= rows) {
          ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid local epilogue inverse index"); });
          return;
        }
        auto weighted = terms[term * TILE];
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (source < active) {
          DataCopy(input, input_[source * WIDTH + column], TILE);
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
          Cast(weighted, input, RoundMode::CAST_NONE, TILE);
        } else {
          // Reproduce the producer's +0, including 0 * negative/NaN/Inf
          // weights. Never load the uninitialized inactive input row.
          Duplicate(weighted, 0.0f, TILE);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const float weight = weights_.GetValue(route);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PipeBarrier<PIPE_V>();
        Muls(weighted, weighted, weight, TILE);
        PipeBarrier<PIPE_V>();
      }
      auto result = terms;
      if constexpr (ORDER == 1) {
        Duplicate(sum, 0.0f, TILE);
        PipeBarrier<PIPE_V>();
        for (uint32_t term = 0; term < TOP_K; ++term) {
          Add(sum, sum, terms[term * TILE], TILE);
          PipeBarrier<PIPE_V>();
        }
        result = sum;
      } else if constexpr (ORDER == 2) {
        // Keep nodes at their original offsets; no overlapping tensor copy.
        for (uint32_t stride = 1; stride < TOP_K; stride *= 2) {
          for (uint32_t term = 0; term + stride < TOP_K; term += 2 * stride) {
            Add(terms[term * TILE], terms[term * TILE], terms[(term + stride) * TILE], TILE);
          }
          PipeBarrier<PIPE_V>();
        }
      }
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output_[outputRow * WIDTH + column], result, TILE);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputBuffer_, termsBuffer_, sumBuffer_;
  GlobalTensor<half> input_;
  GlobalTensor<int64_t> inverse_, ends_;
  GlobalTensor<float> output_, weights_;
};
}  // namespace

#define LOCAL_EPILOGUE_ENTRY(NAME, ORDER)                                                                    \
  extern "C" __global__ __aicore__ void NAME(GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights, GM_ADDR ends, \
                                             GM_ADDR output, GM_ADDR config) {                               \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);                                                            \
    InitSocState();                                                                                          \
    const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];                                       \
    if (rows <= 0 || rows > MAX_ROUTES || rows % TOP_K) return;                                              \
    LocalEpilogue<ORDER> op;                                                                                 \
    op.Run(routed, inverse, weights, ends, output, rows);                                                    \
  }

LOCAL_EPILOGUE_ENTRY(qwen_local_route_finalize_v1, 0)
LOCAL_EPILOGUE_ENTRY(qwen_local_route_reduce_sequential_v1, 1)
LOCAL_EPILOGUE_ENTRY(qwen_local_route_reduce_pairwise_v1, 2)
