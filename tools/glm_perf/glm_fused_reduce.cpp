// SPDX-License-Identifier: Apache-2.0
// Stable FP32 reduction of bulk-prefill native INT4 down projections.
#include "kernel_operator.h"
#include "glm_fused_reduce_schedule.h"
namespace {
using namespace AscendC;
constexpr uint32_t N = 128;
#if defined(GLM_NATIVE_ROUTE_COLUMNS) && !defined(GLM_FP16_ROUTE_WORKSPACE)
  #error "native route columns require the paired FP16 workspace producer"
#endif
#if defined(GLM_PREFILL_REDUCE_META_CACHE) && !defined(GLM_FP16_ROUTE_WORKSPACE)
  #error "cached reducer metadata requires the FP16 route workspace"
#endif
class Reduce {
 public:
  __aicore__ inline void Run(GM_ADDR workspace, GM_ADDR ranks, GM_ADDR ends, GM_ADDR output, const int64_t* config,
                             GM_ADDR order = nullptr, GM_ADDR weights = nullptr) {
    const int64_t rows = config[0], experts = config[1], width = config[2], tokens = config[6], topK = config[7];
#ifdef GLM_FP16_ROUTE_WORKSPACE
    halfInput_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));
    order_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(order));
    weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    pipe_.InitBuffer(halfStorage_, N * sizeof(half));
#else
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
#endif
    ranks_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranks));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(storage_, 2 * N * sizeof(float));
#ifdef GLM_PREFILL_REDUCE_META_CACHE
    pipe_.InitBuffer(metadataStorage_, GlmFusedReduceSchedule::METADATA_BYTES);
#endif
#ifdef GLM_NATIVE_ROUTE_COLUMNS
    pipe_.InitBuffer(offsetStorage_, N * sizeof(uint32_t));
    auto offsets = offsetStorage_.Get<uint32_t>();
    for (uint32_t channel = 0; channel < N; ++channel)
      offsets.SetValue(channel, (channel / 2 + (channel % 2 ? N / 2 : 0)) * sizeof(float));
    PipeBarrier<PIPE_ALL>();
#endif
    const int64_t localRows = ends_.GetValue(experts - 1), tiles = width / N;
    if (localRows < 0 || localRows > rows) return;
#ifdef GLM_PREFILL_REDUCE_META_CACHE
    // Every core owns the same number of complete tokens. Each token's routes
    // and weights are read once, then reused across all of its output tiles.
    // Uneven/small batches retain the balanced column-task schedule below.
    if (tokens >= GlmFusedReduceSchedule::MIN_TOKENS && tokens % GetBlockNum() == 0 &&
        topK <= GlmFusedReduceSchedule::MAX_TOP_K) {
      for (int64_t token = GetBlockIdx(); token < tokens; token += GetBlockNum()) {
        PrepareToken(token, topK, localRows);
        for (int64_t column = 0; column < width; column += N) ReduceTile(token, column, width, topK, localRows, true);
      }
      return;
    }
#endif
    for (int64_t task = GetBlockIdx(); task < tokens * tiles; task += GetBlockNum()) {
      const int64_t token = task / tiles, column = (task % tiles) * N;
      ReduceTile(token, column, width, topK, localRows, false);
    }
  }

 private:
  __aicore__ inline void ReduceTile(int64_t token, int64_t column, int64_t width, int64_t topK, int64_t localRows,
                                    bool cached) {
    auto accum = storage_.Get<float>(), value = accum[N];
    Duplicate(accum, 0.0f, N);
    PipeBarrier<PIPE_ALL>();
    // ranks are sorted by expert's stable route position within each token.
    // Never read the uninitialized peer/zero-weight suffix of the workspace.
    for (int64_t slot = 0; slot < topK; ++slot) {
      const int64_t row = Rank(token, slot, topK, cached);
      if (row < 0 || row >= localRows) continue;
#ifdef GLM_FP16_ROUTE_WORKSPACE
      DataCopy(halfStorage_.Get<half>(), halfInput_[row * width + column], N);
      PipeBarrier<PIPE_ALL>();
      Cast(value, halfStorage_.Get<half>(), RoundMode::CAST_NONE, N);
      PipeBarrier<PIPE_V>();
      Muls(value, value, Weight(row, slot, cached), N);
#else
      DataCopy(value, input_[row * width + column], N);
#endif
      PipeBarrier<PIPE_ALL>();
      Add(accum, accum, value, N);
      PipeBarrier<PIPE_ALL>();
    }
#ifdef GLM_NATIVE_ROUTE_COLUMNS
    auto offsets = offsetStorage_.Get<uint32_t>();
    Gather(value, accum, offsets, static_cast<uint32_t>(0), N);
    PipeBarrier<PIPE_ALL>();
    DataCopy(output_[token * width + column], value, N);
#else
    DataCopy(output_[token * width + column], accum, N);
#endif
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline int64_t Rank(int64_t token, int64_t slot, int64_t topK, bool cached) {
#ifdef GLM_PREFILL_REDUCE_META_CACHE
    if (cached) return metadataStorage_.Get<int32_t>().GetValue(slot);
#endif
    return ranks_.GetValue(token * topK + slot);
  }
#ifdef GLM_FP16_ROUTE_WORKSPACE
  __aicore__ inline float Weight(int64_t row, int64_t slot, bool cached) {
  #ifdef GLM_PREFILL_REDUCE_META_CACHE
    if (cached) return metadataStorage_.Get<float>()[GlmFusedReduceSchedule::MAX_TOP_K].GetValue(slot);
  #endif
    return weights_.GetValue(order_.GetValue(row));
  }
#endif
#ifdef GLM_PREFILL_REDUCE_META_CACHE
  __aicore__ inline void PrepareToken(int64_t token, int64_t topK, int64_t localRows) {
    auto cachedRanks = metadataStorage_.Get<int32_t>();
    auto cachedWeights = metadataStorage_.Get<float>()[GlmFusedReduceSchedule::MAX_TOP_K];
    for (int64_t slot = 0; slot < topK; ++slot) {
      const int32_t row = ranks_.GetValue(token * topK + slot);
      cachedRanks.SetValue(slot, row);
      cachedWeights.SetValue(slot, row >= 0 && row < localRows ? weights_.GetValue(order_.GetValue(row)) : 0.0f);
    }
    PipeBarrier<PIPE_ALL>();
  }
  TBuf<TPosition::VECCALC> metadataStorage_;
#endif
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
#ifdef GLM_NATIVE_ROUTE_COLUMNS
  TBuf<TPosition::VECCALC> offsetStorage_;
#endif
  GlobalTensor<float> input_, output_;
  GlobalTensor<int32_t> ranks_;
  GlobalTensor<int64_t> ends_;
#ifdef GLM_FP16_ROUTE_WORKSPACE
  TBuf<TPosition::VECCALC> halfStorage_;
  GlobalTensor<half> halfInput_;
  GlobalTensor<int64_t> order_;
  GlobalTensor<float> weights_;
#endif
};
}  // namespace
#ifdef GLM_FP16_ROUTE_WORKSPACE
extern "C" __global__ __aicore__ void glm_fused_reduce_half_v1(GM_ADDR workspace, GM_ADDR ranks, GM_ADDR ends,
                                                               GM_ADDR output, GM_ADDR tiling, GM_ADDR order,
                                                               GM_ADDR weights) {
#else
extern "C" __global__ __aicore__ void glm_fused_reduce_v1(GM_ADDR workspace, GM_ADDR ranks, GM_ADDR ends,
                                                          GM_ADDR output, GM_ADDR tiling) {
#endif
  AscendC::InitSocState();
  int64_t config[8];
  const auto source = reinterpret_cast<__gm__ int64_t*>(tiling);
  for (unsigned i = 0; i < 8; ++i) config[i] = source[i];
  if (config[0] <= 0 || config[0] > 65536 || config[1] <= 0 || config[2] <= 0 || config[2] % N || config[6] <= 16 ||
      config[7] <= 0 || config[6] * config[7] != config[0])
    return;
  Reduce operation;
  operation.Run(workspace, ranks, ends, output, config
#ifdef GLM_FP16_ROUTE_WORKSPACE
                ,
                order, weights
#endif
  );
}
