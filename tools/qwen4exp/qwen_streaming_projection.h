// SPDX-License-Identifier: Apache-2.0
// One integrated producer/Cube/consumer schedule for grouped gate/up and down.
#ifndef QWEN_STREAMING_PROJECTION_H
#define QWEN_STREAMING_PROJECTION_H
#include "kernel_operator.h"
#include "qwen_streaming_contract.h"
#include "qwen_streaming_operands.h"

namespace qwen_streaming {
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
    // Reserve the complete contract, including the future paired epilogue.
    // No arena alias changes when an expert's last row tile shrinks.
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
      const int64_t end = ends_.GetValue(expert);
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

  // Startup issues group 0 without a prior consumer. Thereafter Cube j and
  // vector consumption j-1 own disjoint slots. CO1 readback is completed before
  // another Cube write; consumption stays in ascending G128 order through drain.
  __aicore__ inline void Run(int64_t row, uint32_t liveRows) {
    Duplicate(Ub<float>(ACCUMULATOR_OFFSET), 0.0f, M * N);
    int64_t previous = -1;
    for (uint32_t group = 0; group < groups_; ++group) {
      const uint32_t slot = Slot(group);
      producer_.Produce(slot, row, group, liveRows);
      StageMetadata(slot, group);
      IssueCube(slot);
      if (previous >= 0) Consume(Slot(previous), liveRows);
      ReadBack(slot);
      previous = group;
    }
    Consume(Slot(previous), liveRows);
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

  __aicore__ inline void Consume(uint32_t slot, uint32_t liveRows) {
    auto result = Ub<float>(FLOAT_PRODUCT_OFFSET + slot * FLOAT_PRODUCT_SLOT_BYTES);
    auto raw = Ub<int32_t>(RAW_PRODUCT_OFFSET + slot * RAW_PRODUCT_SLOT_BYTES);
    auto accumulator = Ub<float>(ACCUMULATOR_OFFSET);
    auto metadata = Ub<float>(METADATA_STAGE_OFFSET + slot * METADATA_STAGE_SLOT_BYTES);
    auto activation = Ub<float>(ACTIVATION_METADATA_OFFSET + slot * ACTIVATION_METADATA_SLOT_BYTES);
    auto sw = metadata;
    auto zw = metadata[N];
    auto ws = metadata[2 * N];
    auto xs = activation;
    auto sums = activation[M * LANES];
    constexpr uint8_t PRODUCT_STRIDE = 2 * M * BLOCK / LANES;
    constexpr uint8_t ACCUMULATOR_STRIDE = M * BLOCK / LANES;
    constexpr uint8_t REPEATS = N / BLOCK;
    Cast(result, raw, RoundMode::CAST_NONE, 2 * M * N);
    Muls(ws, ws, 8.0f, N);
    PipeBarrier<PIPE_V>();
    // Each row is vectorized across all N16 strips. Independent integer dots
    // widen exactly; the FP32 correction sequence matches the existing schedule.
    auto high = result[M * BLOCK];
    for (uint32_t row = 0; row < liveRows; ++row)
      Muls(high[row * BLOCK], high[row * BLOCK], 16.0f, BLOCK, REPEATS, {1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Add(result[row * BLOCK], result[row * BLOCK], high[row * BLOCK], BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Mul(high[row * BLOCK], zw, sums[row * LANES], BLOCK, REPEATS, {1, 1, 0, PRODUCT_STRIDE, BLOCK / LANES, 0});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Sub(result[row * BLOCK], result[row * BLOCK], high[row * BLOCK], BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Add(result[row * BLOCK], result[row * BLOCK], ws, BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, BLOCK / LANES});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Mul(result[row * BLOCK], result[row * BLOCK], sw, BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, BLOCK / LANES});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Mul(result[row * BLOCK], result[row * BLOCK], xs[row * LANES], BLOCK, REPEATS,
          {1, 1, 0, PRODUCT_STRIDE, PRODUCT_STRIDE, 0});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < liveRows; ++row)
      Add(accumulator[row * BLOCK], accumulator[row * BLOCK], result[row * BLOCK], BLOCK, REPEATS,
          {1, 1, 1, ACCUMULATOR_STRIDE, ACCUMULATOR_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
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
    auto cache = Ub<half>(WEIGHT_METADATA_OFFSET);
    const int64_t first = (expert * n_ + tile * N) * groups_;
    DataCopy(cache, scales_[first], N * groups_);
    DataCopy(cache[N * MAX_GROUPS], offsets_[first], N * groups_);
    DataCopy(cache[2 * N * MAX_GROUPS], weightSums_[first], N * groups_);
    SetFlag<HardEvent::MTE2_V>(CONTROL_EVENT);
    WaitFlag<HardEvent::MTE2_V>(CONTROL_EVENT);
  }

  __aicore__ inline void StageMetadata(uint32_t slot, uint32_t group) {
    auto cache = Ub<half>(WEIGHT_METADATA_OFFSET);
    auto stage = Ub<float>(METADATA_STAGE_OFFSET + slot * METADATA_STAGE_SLOT_BYTES);
    for (uint32_t bank = 0; bank < 3; ++bank)
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
        Cast(stage[bank * N + nb * BLOCK], cache[bank * N * MAX_GROUPS + (nb * groups_ + group) * BLOCK],
             RoundMode::CAST_NONE, BLOCK);
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
  OperandProducer producer_;
  GlobalTensor<int8_t> low_, high_, codes_;
  GlobalTensor<float> xs_, sums_;
  GlobalTensor<half> scales_, offsets_, weightSums_, output_;
  GlobalTensor<int64_t> ends_;
  int64_t rows_ = 0, experts_ = 0, n_ = 0, k_ = 0, groups_ = 0;
  int64_t firstTile_ = 0, tileCount_ = 0, outputWidth_ = 0;
};
}  // namespace qwen_streaming
#endif
