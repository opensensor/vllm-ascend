// SPDX-License-Identifier: Apache-2.0
// New channel-scale contract: accumulate INT32 across K, read back once/group.
#include "kernel_operator.h"
#include "glm_channel_dma.h"
#include "glm_channel_broadcast.h"
#ifdef GLM_CHANNEL_INTEGER_RECONSTRUCT
  #include "glm_channel_integer.h"
#endif
#ifdef GLM_CHANNEL_CACHE_SOURCE_ROWS
  #include "glm_channel_row_cache.h"
#endif
namespace {
using namespace AscendC;
#ifndef GLM_CHANNEL_ROWS
  #define GLM_CHANNEL_ROWS 32
#endif
constexpr uint32_t ROWS = GLM_CHANNEL_ROWS, N = 128, K = 256, K_FRAGMENT = 64, NZ_N = 16, LANES = 8;
constexpr uint32_t A_BYTES = ROWS * K / 2, B_BYTES = N * K / 2, ELEMENTS = ROWS * N;
class Projection {
 public:
  __aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR sums,
                             GM_ADDR ends, GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets, GM_ADDR output,
                             const __gm__ int64_t* config) {
    if (!Init(low, high, xs, codes, scales, sums, ends, order, sortedKeys, offsets, output, config)) return;
    const uint32_t maxRows = bits_ == 8 ? ROWS / 2 : ROWS;
    for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
      int64_t first = 0;
      const int64_t valid = ends_.GetValue(experts_ - 1);
      for (int64_t expert = 0; expert < experts_ && first < valid; ++expert) {
        if (tokens_ <= 16) expert = static_cast<int64_t>(sortedKeys_.GetValue(first));
        if (expert < 0 || expert >= experts_) return;
        const int64_t end = ends_.GetValue(expert);
        if (end < first || end > tokens_ * topK_) return;
        for (int64_t row = first; row < end; row += maxRows) {
          const uint32_t count = end - row < maxRows ? end - row : maxRows;
          ProjectBlock(row, count, expert, tile);
          const DataCopyParams store{static_cast<uint16_t>(count), N * sizeof(half) / 32, 0,
                                     static_cast<uint16_t>((n_ - N) * sizeof(half) / 32)};
          DataCopy(output_[row * n_ + tile * N], outputUb_.Get<half>(), store);
          GlmChannelDma::Fence();
        }
        first = end;
      }
    }
  }

 protected:
  __aicore__ inline bool Init(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR sums,
                              GM_ADDR ends, GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets, GM_ADDR output,
                              const __gm__ int64_t* config) {
    tokens_ = config[0];
    topK_ = config[1];
    experts_ = config[2];
    n_ = config[3];
    k_ = config[4];
    group_ = config[5];
    bits_ = config[6];
    originalTokens_ = config[7] != 0;
    if (tokens_ < 1 || topK_ < 1 || experts_ < 1 || n_ % N || k_ % K || group_ < K || group_ > k_ || group_ % K ||
        k_ % group_ || (bits_ != 4 && bits_ != 8))
      return false;
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    codes_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(codes));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    sums_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sums));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ends));
    sortedKeys_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedKeys));
    order_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(order));
    offsets_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    pipe_.InitBuffer(aUb_, A_BYTES);
    pipe_.InitBuffer(a1_, A_BYTES);
    pipe_.InitBuffer(a2_, A_BYTES);
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    pipe_.InitBuffer(b1_, 2 * B_BYTES);
    pipe_.InitBuffer(b2_, 2 * B_BYTES);
#else
    pipe_.InitBuffer(b1_, B_BYTES);
    pipe_.InitBuffer(b2_, B_BYTES);
#endif
    pipe_.InitBuffer(c_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(raw_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(indices_, ELEMENTS * sizeof(uint32_t));
    pipe_.InitBuffer(products_, ELEMENTS * sizeof(float));
    pipe_.InitBuffer(accum_, ELEMENTS * sizeof(float));
    pipe_.InitBuffer(weights_, 2 * N * sizeof(float));
    pipe_.InitBuffer(sx_, ROWS * LANES * sizeof(float));
    pipe_.InitBuffer(outputUb_, ELEMENTS * sizeof(half));
#ifdef GLM_CHANNEL_CACHE_SOURCE_ROWS
    pipe_.InitBuffer(sourceRows_, ROWS * sizeof(int32_t));
#endif
    return true;
  }
  __aicore__ inline void ProjectBlock(int64_t row, uint32_t count, int64_t expert, int64_t tile) {
#ifdef GLM_CHANNEL_CACHE_SOURCE_ROWS
    if (originalTokens_) {
      GlmChannelRowCache::Prepare(order_, sourceRows_.Get<int32_t>(), row, count, topK_);
      GlmChannelDma::Fence();
    }
#endif
    const uint32_t productRows = ((count * (bits_ == 8 ? 2 : 1) + NZ_N - 1) / NZ_N) * NZ_N;
    const uint32_t groups = k_ / group_;
    Duplicate(accum_.Get<float>(), 0.0f, count * N);
    PipeBarrier<PIPE_V>();
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    // Independent two-slot B pipeline. A/readback storage remains single.
    for (uint32_t buffer = 0; buffer < 2; ++buffer) {
      SetFlag<HardEvent::MTE1_MTE2>(buffer);
      SetFlag<HardEvent::M_MTE1>(buffer);
    }
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    PrefetchWeight(expert, tile, 0, 0);
#endif
    for (uint32_t group = 0; group < groups; ++group) {
      for (uint32_t k0 = group * group_; k0 < (group + 1) * group_; k0 += K) {
        LoadActivation(row, count, productRows, k0);
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
        const uint32_t buffer = (k0 / K) % 2;
        WaitFlag<HardEvent::MTE2_MTE1>(buffer);
        WaitFlag<HardEvent::M_MTE1>(buffer);
#else
        const int64_t codeOffset = ((expert * (n_ / N) + tile) * (k_ / K) + k0 / K) * B_BYTES;
        DataCopy(b1_.Get<int8_t>(), codes_[codeOffset], B_BYTES);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#endif
        LoadData2DParams load;
        load.repeatTimes = (K / K_FRAGMENT) * N / NZ_N;
        load.srcStride = 1;
        load.ifTranspose = false;
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
        LoadData(b2_.Get<int8_t>()[buffer * B_BYTES].ReinterpretCast<int4b_t>(),
                 b1_.Get<int8_t>()[buffer * B_BYTES].ReinterpretCast<int4b_t>(), load);
        SetFlag<HardEvent::MTE1_MTE2>(buffer);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
#else
        LoadData(b2_.Get<int8_t>().ReinterpretCast<int4b_t>(), b1_.Get<int8_t>().ReinterpretCast<int4b_t>(), load);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
#endif
        MmadParams mm;
        mm.m = productRows;
        mm.n = N;
        mm.k = K;
        mm.cmatrixInitVal = k0 == group * group_;
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
        Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>().ReinterpretCast<int4b_t>(),
             b2_.Get<int8_t>()[buffer * B_BYTES].ReinterpretCast<int4b_t>(), mm);
        SetFlag<HardEvent::M_MTE1>(buffer);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        // Submit next GM -> L1 weight movement while this Cube consumes B2.
        if (k0 + K < k_) PrefetchWeight(expert, tile, k0 + K, 1 - buffer);
#else
        Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>().ReinterpretCast<int4b_t>(),
             b2_.Get<int8_t>().ReinterpretCast<int4b_t>(), mm);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        // Conservative operand lifetime fence. No cast, scale, FP32 sum,
        // or CO1 readback is performed in this K loop.
        GlmChannelDma::Fence();
#endif
      }
      Readback(row, count, productRows, expert, tile, group);
    }
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
    for (uint32_t buffer = 0; buffer < 2; ++buffer) {
      WaitFlag<HardEvent::MTE1_MTE2>(buffer);
      WaitFlag<HardEvent::M_MTE1>(buffer);
    }
#endif
    Cast(outputUb_.Get<half>(), accum_.Get<float>(), RoundMode::CAST_NONE, count * N);
    GlmChannelDma::Fence();
  }

  __aicore__ inline int64_t SourceRow(int64_t row) {
    if (!originalTokens_) return row;
    const int64_t position = order_.GetValue(row);
    if (topK_ == 8) return position >> 3;
    if (topK_ == 4) return position >> 2;
    if (topK_ == 2) return position >> 1;
    if (topK_ == 1) return position;
    return position / topK_;
  }
  __aicore__ inline int64_t RowAddress(int64_t row, uint32_t m) {
#ifdef GLM_CHANNEL_TOKEN_PREFILL
    if (indirectRows_) return tokenRowsUb_.Get<int32_t>().GetValue(indirectStart_ + m);
#endif
#ifdef GLM_CHANNEL_CACHE_SOURCE_ROWS
    return originalTokens_ ? sourceRows_.Get<int32_t>().GetValue(m) : row + m;
#else
    return SourceRow(row + m);
#endif
  }
  __aicore__ inline void LoadActivation(int64_t row, uint32_t count, uint32_t productRows, uint32_t k0) {
    auto packed = aUb_.Get<int8_t>();
    Duplicate(packed.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), productRows * K / 4);
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
#else
    GlmChannelDma::Fence();
#endif
    for (uint32_t m = 0; m < count; ++m) {
      const int64_t sourceRow = RowAddress(row, m) * (k_ / 2) + k0 / 2;
      for (uint32_t part = 0; part < K / K_FRAGMENT; ++part) {
        const int64_t source = sourceRow + part * K_FRAGMENT / 2;
        const uint32_t target = (part * productRows + m * (bits_ == 8 ? 2 : 1)) * K_FRAGMENT / 2;
        DataCopy(packed[target], low_[source], K_FRAGMENT / 2);
        if (bits_ == 8) DataCopy(packed[target + K_FRAGMENT / 2], high_[source], K_FRAGMENT / 2);
      }
    }
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
#else
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
#endif
    DataCopy(a1_.Get<int8_t>(), packed, productRows * K / 2);
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
#else
    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
#endif
    LoadData2DParams load;
    load.repeatTimes = (K / K_FRAGMENT) * productRows / NZ_N;
    load.srcStride = 1;
    load.ifTranspose = false;
    LoadData(a2_.Get<int8_t>().ReinterpretCast<int4b_t>(), a1_.Get<int8_t>().ReinterpretCast<int4b_t>(), load);
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
    // A1 and UB must be free before a later call overwrites them.
    SetFlag<HardEvent::MTE1_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_V>(EVENT_ID0);
#else
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_V>(EVENT_ID0);
#endif
  }
#ifdef GLM_CHANNEL_STREAM_WEIGHTS
  __aicore__ inline void PrefetchWeight(int64_t expert, int64_t tile, uint32_t k0, uint32_t buffer) {
    WaitFlag<HardEvent::MTE1_MTE2>(buffer);
    const int64_t offset = ((expert * (n_ / N) + tile) * (k_ / K) + k0 / K) * B_BYTES;
    DataCopy(b1_.Get<int8_t>()[buffer * B_BYTES], codes_[offset], B_BYTES);
    SetFlag<HardEvent::MTE2_MTE1>(buffer);
  }
#endif
  __aicore__ inline void Readback(int64_t row, uint32_t count, uint32_t productRows, int64_t expert, int64_t tile,
                                  uint32_t group) {
    const DataCopyParams copy{N / NZ_N, static_cast<uint16_t>(productRows / NZ_N), 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    SetFlag<HardEvent::M_V>(EVENT_ID0);
    WaitFlag<HardEvent::M_V>(EVENT_ID0);
    DataCopy(raw_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
    GlmChannelDma::Fence();
    auto low = products_.Get<float>();
    auto high = low[ELEMENTS / 2];
    const uint32_t slot = (productRows / NZ_N - 1) * 4 + (bits_ == 8 ? 2 : 0);
    GlmChannelDma::CopyIn(indices_.Get<uint32_t>(), offsets_[slot * ELEMENTS], count * N);
    GlmChannelDma::Fence();
    Gather(low.ReinterpretCast<int32_t>(), raw_.Get<int32_t>(), indices_.Get<uint32_t>(), static_cast<uint32_t>(0),
           count * N);
    PipeBarrier<PIPE_V>();
    const int64_t scaleOffset = ((expert * (n_ / N) + tile) * (k_ / group_) + group) * N;
    auto weights = weights_.Get<float>();
    auto correction = weights[N];
    GlmChannelDma::CopyIn(weights, scales_[scaleOffset], N);
    if (bits_ == 8) {
      GlmChannelDma::CopyIn(indices_.Get<uint32_t>(), offsets_[(slot + 1) * ELEMENTS], count * N);
      GlmChannelDma::Fence();
      Gather(high.ReinterpretCast<int32_t>(), raw_.Get<int32_t>(), indices_.Get<uint32_t>(), static_cast<uint32_t>(0),
             count * N);
      GlmChannelDma::CopyIn(correction.ReinterpretCast<int32_t>(), sums_[scaleOffset], N);
      GlmChannelDma::Fence();
#ifdef GLM_CHANNEL_INTEGER_RECONSTRUCT
      GlmChannelInteger::ReconstructA8(low.ReinterpretCast<int32_t>(), high.ReinterpretCast<int32_t>(),
                                       correction.ReinterpretCast<int32_t>(), count);
      Cast(low, low.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count * N);
#else
      Cast(low, low.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count * N);
      Cast(high, high.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count * N);
      Cast(correction, correction.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, N);
      PipeBarrier<PIPE_V>();
      Muls(high, high, 16.0f, count * N);
      Muls(correction, correction, 8.0f, N);
      PipeBarrier<PIPE_V>();
      Add(low, low, high, count * N);
#endif
    } else {
      Cast(low, low.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count * N);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t m = 0; m < count; ++m)
      GlmChannelDma::CopyIn(sx_.Get<float>()[m * LANES], xs_[(RowAddress(row, m) * (k_ / group_) + group) * LANES],
                            LANES);
    GlmChannelDma::Fence();
#ifdef GLM_CHANNEL_INTEGER_RECONSTRUCT
    // A8 bias has already been added exactly before the product conversion.
    GlmChannelBroadcast::ScaleRows(low, weights, correction, sx_.Get<float>(), count, 4);
#else
    GlmChannelBroadcast::ScaleRows(low, weights, correction, sx_.Get<float>(), count, bits_);
#endif
    Add(accum_.Get<float>(), accum_.Get<float>(), low, count * N);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_M>(EVENT_ID0);
    WaitFlag<HardEvent::V_M>(EVENT_ID0);
  }
  TPipe pipe_;
#ifdef GLM_CHANNEL_TOKEN_PREFILL
  TBuf<TPosition::VECCALC> tokenRowsUb_;
  bool indirectRows_ = false;
  uint32_t indirectStart_ = 0;
#endif
  TBuf<TPosition::VECCALC> aUb_, raw_, indices_, products_, accum_, weights_, sx_, outputUb_;
#ifdef GLM_CHANNEL_CACHE_SOURCE_ROWS
  TBuf<TPosition::VECCALC> sourceRows_;
#endif
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  GlobalTensor<int8_t> low_, high_, codes_;
  GlobalTensor<float> xs_, scales_;
  GlobalTensor<int32_t> sortedKeys_;
  GlobalTensor<int32_t> sums_;
  GlobalTensor<int32_t> ends_;
  GlobalTensor<int64_t> order_;
  GlobalTensor<uint32_t> offsets_;
  GlobalTensor<half> output_;
  int64_t tokens_, topK_, experts_, n_, k_, group_, bits_;
  bool originalTokens_;
};
}  // namespace
#ifndef GLM_CHANNEL_FUSED_GATE_UP
extern "C" __global__ __aicore__ void glm_channel_projection_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                                GM_ADDR scales, GM_ADDR sums, GM_ADDR ends,
                                                                GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR offsets,
                                                                GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  Projection operation;
  operation.Run(low, high, xs, codes, scales, sums, ends, order, sortedKeys, offsets, output,
                reinterpret_cast<__gm__ int64_t*>(config));
}

#endif
