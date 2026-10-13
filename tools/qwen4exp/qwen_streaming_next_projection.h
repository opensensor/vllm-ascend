// SPDX-License-Identifier: Apache-2.0
// One integrated producer/Cube/consumer schedule for grouped gate/up and down.
#ifndef QWEN_STREAMING_NEXT_PROJECTION_H
#define QWEN_STREAMING_NEXT_PROJECTION_H
#include "kernel_operator.h"
#include "qwen_streaming_next_contract.h"
#include "qwen_streaming_next_operands.h"

namespace qwen_streaming_next {
using namespace AscendC;

class Projection {
 public:
  __aicore__ inline void Init(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                              GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends, GM_ADDR output,
                              __gm__ const int64_t* config) {
    rows_ = config[0];
    experts_ = config[1];
    n_ = config[2];
    k_ = config[3];
    groups_ = k_ / GROUP;
    firstTile_ = 0;
    tileCount_ = n_ / N;
    outputWidth_ = n_;
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    sums_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sums));
    codes_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(codes));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
    offsets_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(offset));
    weightSums_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weightSum));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    // Projection-only arena. FP16 output stages one metadata bank only
    // during PrepareExpert, before any row tile or store is live.
    pipe_.InitBuffer(ub_, UB_USED);
    pipe_.InitBuffer(a1_, ACTIVATION_L1_BYTES);
    pipe_.InitBuffer(b1_, RESIDENT_WEIGHT_BYTES);
    pipe_.InitBuffer(a2_, ACTIVATION_L0_BYTES);
    pipe_.InitBuffer(b2_, WEIGHT_L0_BYTES);
    pipe_.InitBuffer(c_, CUBE_PRODUCT_BYTES);
    producer_.Init(low_, high_, xs_, sums_, Ub<int8_t>(PACKED_ACTIVATION_OFFSET), Ub<float>(ACTIVATION_METADATA_OFFSET),
                   a1_.Get<int8_t>(), a2_.Get<int8_t>(), b2_.Get<int8_t>(), b1_.Get<int8_t>(), k_);
  }

  __aicore__ inline void Process() {
    int64_t previousEnd = 0;
    for (int64_t expert = 0; expert < experts_; ++expert) {
      if (expert % END_CACHE_SIZE == 0) {
        const uint32_t count = experts_ - expert < END_CACHE_SIZE ? experts_ - expert : END_CACHE_SIZE;
        const uint32_t aligned = count / 4 * 4;
        SetFlag<HardEvent::S_MTE2>(CONTROL_EVENT);
        WaitFlag<HardEvent::S_MTE2>(CONTROL_EVENT);
        if (aligned) DataCopy(Ub<int64_t>(ENDS_OFFSET), ends_[expert], aligned);
        SetFlag<HardEvent::MTE2_S>(CONTROL_EVENT);
        WaitFlag<HardEvent::MTE2_S>(CONTROL_EVENT);
        for (uint32_t tail = aligned; tail < count; ++tail)
          Ub<int64_t>(ENDS_OFFSET).SetValue(tail, ends_.GetValue(expert + tail));
      }
      const int64_t end = Ub<int64_t>(ENDS_OFFSET).GetValue(expert % END_CACHE_SIZE);
      if (end < previousEnd || end > rows_) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid streaming group boundaries"); });
        return;
      }
      for (int64_t localTile = GetBlockIdx(); localTile < tileCount_; localTile += GetBlockNum()) {
        const int64_t tile = firstTile_ + localTile;
        if (end > previousEnd) {
          PrepareExpert(expert, tile);
          for (int64_t row = previousEnd; row < end; row += M) {
            const uint32_t live = end - row < M ? end - row : M;
            Run(row, live);
            Store(row, live, localTile);
          }
          SetFlag<HardEvent::MTE1_MTE2>(CONTROL_EVENT);
          WaitFlag<HardEvent::MTE1_MTE2>(CONTROL_EVENT);
        }
        if (expert + 1 == experts_) ZeroPeers(end, localTile);
      }
      previousEnd = end;
    }
  }

  // Column windows retain full-bank strides; no weight slicing/copy is required.
  // The caller consumes this bounded FP16 result before reusing its output slot.
  __aicore__ inline void SetColumns(int64_t firstTile, int64_t tileCount) {
    firstTile_ = firstTile;
    tileCount_ = tileCount;
    outputWidth_ = tileCount * N;
  }

  // Readback releases the physical slot before it can be reused two groups
  // later. Three metadata slots separate producer j+1 from consumer j-1.
  // CO1 is single-owner, released only by ReadBack's V_M acknowledgement.
  __aicore__ inline void Run(int64_t row, uint32_t liveRows) {
    Duplicate(Ub<float>(ACCUMULATOR_OFFSET), 0.0f, M * N);
    producer_.Produce(Slot(0), MetadataSlot(0), row, 0, liveRows);
    for (uint32_t group = 0; group < groups_; ++group) {
      IssueCube(Slot(group));
      if (group + 1 < groups_) producer_.Produce(Slot(group + 1), MetadataSlot(group + 1), row, group + 1, liveRows);
      if (group > 0) Consume(group - 1, liveRows);
      ReadBack(Slot(group));
    }
    Consume(groups_ - 1, liveRows);
    PipeBarrier<PIPE_V>();
    Cast(Ub<half>(PROJECTED_OUTPUT_OFFSET), Ub<float>(ACCUMULATOR_OFFSET), RoundMode::CAST_NONE, M * N);
  }

  __aicore__ inline void IssueCube(uint32_t slot) {
    MmadParams mm;
    mm.m = 2 * M;
    mm.n = N;
    mm.k = GROUP;
    mm.cmatrixInitVal = true;
    Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>()[slot * ACTIVATION_L0_SLOT_BYTES].template ReinterpretCast<int4b_t>(),
         b2_.Get<int8_t>()[slot * WEIGHT_L0_SLOT_BYTES].template ReinterpretCast<int4b_t>(), mm);
  }

  __aicore__ inline void ReadBack(uint32_t slot) {
    SetFlag<HardEvent::M_V>(EventId(slot));
    WaitFlag<HardEvent::M_V>(EventId(slot));
    SetFlag<HardEvent::M_MTE1>(EventId(slot));
    WaitFlag<HardEvent::M_MTE1>(EventId(slot));
    DataCopyParams copy{N / BLOCK, 2 * M / BLOCK, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(Ub<int32_t>(RAW_PRODUCT_OFFSET + slot * RAW_PRODUCT_SLOT_BYTES), c_.Get<int32_t>(), copy, enhanced);
    PipeBarrier<PIPE_V>();
    // This acknowledges the read, not merely the MMAD. CO1 has one owner.
    SetFlag<HardEvent::V_M>(EventId(slot));
    WaitFlag<HardEvent::V_M>(EventId(slot));
  }

  __aicore__ inline void Consume(uint32_t group, uint32_t liveRows) {
    const uint32_t slot = Slot(group);
    auto result = Ub<float>(FLOAT_PRODUCT_OFFSET + slot * FLOAT_PRODUCT_SLOT_BYTES);
    auto raw = Ub<int32_t>(RAW_PRODUCT_OFFSET + slot * RAW_PRODUCT_SLOT_BYTES);
    auto accumulator = Ub<float>(ACCUMULATOR_OFFSET);
    auto metadata = Ub<float>(WEIGHT_METADATA_OFFSET);
    auto activation = Ub<float>(ACTIVATION_METADATA_OFFSET + MetadataSlot(group) * ACTIVATION_METADATA_SLOT_BYTES);
    auto sw = metadata[group * BLOCK];
    auto zw = metadata[N * MAX_GROUPS + group * BLOCK];
    auto ws = metadata[2 * N * MAX_GROUPS + group * BLOCK];
    auto xs = activation;
    auto sums = activation[M * LANES];
    constexpr uint8_t PRODUCT_STRIDE = 2 * M * BLOCK / LANES;
    constexpr uint8_t ACCUMULATOR_STRIDE = M * BLOCK / LANES;
    constexpr uint8_t REPEATS = N / BLOCK;
    const uint8_t metadataStride = groups_ * BLOCK / LANES;
    Cast(result, raw, RoundMode::CAST_NONE, 2 * M * N);
    PipeBarrier<PIPE_V>();
    auto high = result[M * BLOCK];
    if (liveRows <= N / BLOCK) {
      // Each row is vectorized across all N16 strips. Independent integer dots
      // widen exactly; the FP32 correction sequence matches the existing schedule.
      for (uint32_t row = 0; row < liveRows; ++row)
        Muls(high[row * BLOCK], high[row * BLOCK], 16.0f, BLOCK, REPEATS, {1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Add(result[row * BLOCK], result[row * BLOCK], high[row * BLOCK], BLOCK, REPEATS,
            {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, PRODUCT_STRIDE});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Mul(high[row * BLOCK], zw, sums[row * LANES], BLOCK, REPEATS, {1, 1, 0, PRODUCT_STRIDE, metadataStride, 0});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Sub(result[row * BLOCK], result[row * BLOCK], high[row * BLOCK], BLOCK, REPEATS,
            {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, PRODUCT_STRIDE});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Add(result[row * BLOCK], result[row * BLOCK], ws, BLOCK, REPEATS,
            {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, metadataStride});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Mul(result[row * BLOCK], result[row * BLOCK], sw, BLOCK, REPEATS,
            {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, metadataStride});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Mul(result[row * BLOCK], result[row * BLOCK], xs[row * LANES], BLOCK, REPEATS,
            {1, 1, 0, PRODUCT_STRIDE, PRODUCT_STRIDE, 0});
      PipeBarrier<PIPE_V>();
      for (uint32_t row = 0; row < liveRows; ++row)
        Add(accumulator[row * BLOCK], accumulator[row * BLOCK], result[row * BLOCK], BLOCK, REPEATS,
            {1, 1, 1, ACCUMULATOR_STRIDE, ACCUMULATOR_STRIDE, PRODUCT_STRIDE});
      PipeBarrier<PIPE_V>();
    } else {
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Muls(high[nb * 2 * M * BLOCK], high[nb * 2 * M * BLOCK], 16.0f, BLOCK, liveRows, {1, 1, 2, 2});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Add(result[nb * 2 * M * BLOCK], result[nb * 2 * M * BLOCK], high[nb * 2 * M * BLOCK], BLOCK, liveRows,
            {1, 1, 1, 2, 2, 2});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Mul(high[nb * 2 * M * BLOCK], zw[nb * groups_ * BLOCK], sums, BLOCK, liveRows, {1, 1, 0, 2, 0, 1});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Sub(result[nb * 2 * M * BLOCK], result[nb * 2 * M * BLOCK], high[nb * 2 * M * BLOCK], BLOCK, liveRows,
            {1, 1, 1, 2, 2, 2});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Add(result[nb * 2 * M * BLOCK], result[nb * 2 * M * BLOCK], ws[nb * groups_ * BLOCK], BLOCK, liveRows,
            {1, 1, 1, 2, 2, 0});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Mul(result[nb * 2 * M * BLOCK], result[nb * 2 * M * BLOCK], sw[nb * groups_ * BLOCK], BLOCK, liveRows,
            {1, 1, 1, 2, 2, 0});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Mul(result[nb * 2 * M * BLOCK], result[nb * 2 * M * BLOCK], xs, BLOCK, liveRows, {1, 1, 0, 2, 2, 1});
      PipeBarrier<PIPE_V>();
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Add(accumulator[nb * M * BLOCK], accumulator[nb * M * BLOCK], result[nb * 2 * M * BLOCK], BLOCK, liveRows,
            {1, 1, 1, 2, 2, 2});
      PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE2>(EventId(slot));
    WaitFlag<HardEvent::V_MTE2>(EventId(slot));
  }

 private:
  template <typename T>
  __aicore__ inline LocalTensor<T> Ub(uint32_t offset) {
    return ub_.Get<T>()[offset / sizeof(T)];
  }

  __aicore__ inline void PrepareExpert(int64_t expert, int64_t tile) {
    SetFlag<HardEvent::V_MTE2>(CONTROL_EVENT);
    WaitFlag<HardEvent::V_MTE2>(CONTROL_EVENT);
    DataCopy(b1_.Get<int8_t>(), codes_[(expert * n_ + tile * N) * k_ / 2], N * k_ / 2);
    SetFlag<HardEvent::MTE2_MTE1>(CONTROL_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(CONTROL_EVENT);
    auto cache = Ub<float>(WEIGHT_METADATA_OFFSET);
    auto stage = Ub<half>(PROJECTED_OUTPUT_OFFSET);
    const int64_t first = (expert * n_ + tile * N) * groups_;
    for (uint32_t bank = 0; bank < 3; ++bank) {
      auto source = bank == 0 ? scales_ : (bank == 1 ? offsets_ : weightSums_);
      SetFlag<HardEvent::V_MTE2>(CONTROL_EVENT);
      WaitFlag<HardEvent::V_MTE2>(CONTROL_EVENT);
      DataCopy(stage, source[first], N * groups_);
      SetFlag<HardEvent::MTE2_V>(CONTROL_EVENT);
      WaitFlag<HardEvent::MTE2_V>(CONTROL_EVENT);
      Cast(cache[bank * N * MAX_GROUPS], stage, RoundMode::CAST_NONE, N * groups_);
      PipeBarrier<PIPE_V>();
    }
    Muls(cache[2 * N * MAX_GROUPS], cache[2 * N * MAX_GROUPS], 8.0f, N * groups_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Store(int64_t row, uint32_t liveRows, int64_t tile) {
    SetFlag<HardEvent::V_MTE3>(CONTROL_EVENT);
    WaitFlag<HardEvent::V_MTE3>(CONTROL_EVENT);
    auto out = Ub<half>(PROJECTED_OUTPUT_OFFSET);
    DataCopyParams copy{static_cast<uint16_t>(liveRows), 1, 0, static_cast<uint16_t>(outputWidth_ / BLOCK - 1)};
    for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
      DataCopy(output_[row * outputWidth_ + tile * N + nb * BLOCK], out[nb * M * BLOCK], copy);
    SetFlag<HardEvent::MTE3_V>(CONTROL_EVENT);
    WaitFlag<HardEvent::MTE3_V>(CONTROL_EVENT);
  }

  __aicore__ inline void ZeroPeers(int64_t activeRows, int64_t tile) {
    Duplicate(Ub<half>(PROJECTED_OUTPUT_OFFSET), static_cast<half>(0), M * N);
    for (int64_t row = activeRows; row < rows_; row += M) {
      const uint32_t live = rows_ - row < M ? rows_ - row : M;
      Store(row, live, tile);
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> ub_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  static constexpr uint32_t END_CACHE_SIZE = ENDS_BYTES / sizeof(int64_t);
  OperandProducer producer_;
  GlobalTensor<int8_t> low_, high_, codes_;
  GlobalTensor<float> xs_, sums_;
  GlobalTensor<half> scales_, offsets_, weightSums_, output_;
  GlobalTensor<int64_t> ends_;
  int64_t rows_ = 0, experts_ = 0, n_ = 0, k_ = 0, groups_ = 0;
  int64_t firstTile_ = 0, tileCount_ = 0, outputWidth_ = 0;
};
}  // namespace qwen_streaming_next
#endif
