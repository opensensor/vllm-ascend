// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef NATIVE_INT4_SCHEDULE_H
#define NATIVE_INT4_SCHEDULE_H
#include "kernel_operator.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"

namespace native_int4 {
using namespace AscendC;
// Separate bounded decode/MTP and prefill schedules. A packed activation tile
// feeds N outputs; the full packed K weight tile stays in L1 across M tiles.
template <uint32_t M, uint32_t N = 64>
class Schedule {
  static constexpr uint32_t GROUP = 128, BLOCK = 16, K0 = 64, LANES = 8;
  static constexpr uint32_t A_BYTES = M * GROUP / 2, B_BYTES = N * GROUP / 2;
  static constexpr uint32_t ELEMENTS = M * N, COLUMN_ELEMENTS = M * BLOCK;
  static constexpr uint32_t MAX_K = 2560;
  static constexpr uint32_t END_CACHE_SIZE = 32;

 public:
  __aicore__ inline void Init(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                              GM_ADDR offset, GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y,
                              __gm__ const QwenW4A8Int4KernelTilingData* td) {
    routed_ = td->routed != 0;
    broadcastFactor_ = td->broadcastFactor;
    routeIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ends));
    rows_ = td->numRows;
    experts_ = td->numExperts;
    n_ = td->nDim;
    k_ = td->kDim;
    groups_ = k_ / GROUP;
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    codes_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(codes));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    sums_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sums));
    sw_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
    zw_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(offset));
    ws_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weightSum));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    pipe_.InitBuffer(a1_, 2 * A_BYTES);
    pipe_.InitBuffer(b1_, N * MAX_K / 2);
    pipe_.InitBuffer(a2_, 2 * A_BYTES);
    pipe_.InitBuffer(b2_, 2 * B_BYTES);
    pipe_.InitBuffer(c_, (M <= 32 ? 2 : 1) * ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(packing_, 2 * A_BYTES);
    pipe_.InitBuffer(integers_, (M <= 32 ? 2 : 1) * ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(result_, 3 * ELEMENTS * sizeof(float));
    pipe_.InitBuffer(metadata_, 3 * N * sizeof(half) + 3 * N * sizeof(float));
    pipe_.InitBuffer(activation_, 2 * M * LANES * sizeof(float));
    pipe_.InitBuffer(output_, ELEMENTS * sizeof(half));
    pipe_.InitBuffer(endCache_, END_CACHE_SIZE * sizeof(int64_t));
    pipe_.InitBuffer(routeCache_, MAX_ROUTES * sizeof(int32_t));
  }

  __aicore__ inline void Process() {
    if (routed_) {
      ProcessRoutes();
      return;
    }
    int64_t previousEnd = 0;
    auto cachedEnds = endCache_.Get<int64_t>();
    for (int64_t expert = 0; expert < experts_; ++expert) {
      if (expert % END_CACHE_SIZE == 0) {
        const uint32_t count = Min(END_CACHE_SIZE, experts_ - expert);
        const uint32_t aligned = count / 4 * 4;
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        if (aligned > 0) DataCopy(cachedEnds, ends_[expert], aligned);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        // At most three tail entries; never overread the E-element allocation.
        for (uint32_t tail = aligned; tail < count; ++tail) cachedEnds.SetValue(tail, ends_.GetValue(expert + tail));
      }
      const int64_t begin = previousEnd;
      const int64_t end = cachedEnds.GetValue(expert % END_CACHE_SIZE);
      if (begin < 0 || end < begin || end > rows_) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid native INT4 group boundaries"); });
        return;
      }
      previousEnd = end;
      if (begin == end && expert + 1 != experts_) continue;
      for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
        if (begin < end) {
          // Contiguous packed N=16 strips, each containing every K group.
          DataCopy(b1_.Get<int8_t>(), codes_[(expert * n_ + tile * N) * k_ / 2], N * k_ / 2);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
          for (int64_t row = begin; row < end; row += M) {
            Project(expert, tile, row, Min(M, end - row));
          }
          SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
          WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
        if (expert + 1 == experts_) {
          auto out = output_.Get<half>();
          Duplicate(out, static_cast<half>(0), ELEMENTS);
          SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
          for (int64_t row = end; row < rows_; row += M) Store(out, tile, row, Min(M, rows_ - row));
          SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
      }
    }
  }

 private:
  static constexpr uint32_t MAX_ROUTES = 128;
  __aicore__ inline void ProcessRoutes() {
    int32_t expertIds[MAX_ROUTES];
    uint32_t counts[MAX_ROUTES], groupOfRow[MAX_ROUTES], starts[MAX_ROUTES + 1];
    uint32_t groups = 0;
    auto cachedIds = routeCache_.Get<int32_t>();
    const uint32_t aligned = rows_ / 8 * 8;
    if (aligned > 0) {
      SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
      DataCopy(cachedIds, routeIds_[0], aligned);
      SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    for (uint32_t row = aligned; row < rows_; ++row) cachedIds.SetValue(row, routeIds_.GetValue(row));
    for (uint32_t row = 0; row < rows_; ++row) {
      const int32_t expert = cachedIds.GetValue(row);
      groupOfRow[row] = MAX_ROUTES;
      if (expert < 0 || expert >= experts_) continue;
      uint32_t group = 0;
      while (group < groups && expertIds[group] != expert) ++group;
      if (group == groups) {
        expertIds[group] = expert;
        counts[group] = 0;
        ++groups;
      }
      groupOfRow[row] = group;
      ++counts[group];
    }
    starts[0] = 0;
    for (uint32_t group = 0; group < groups; ++group) {
      starts[group + 1] = starts[group] + counts[group];
      counts[group] = starts[group];
    }
    for (uint32_t row = 0; row < rows_; ++row) {
      sourceRows_[row] = row / broadcastFactor_;
      if (groupOfRow[row] != MAX_ROUTES) routeRows_[counts[groupOfRow[row]]++] = row;
    }
    const uint32_t tiles = n_ / N;
    const uint32_t partitions = GetBlockNum() > tiles && GetBlockNum() % tiles == 0 ? GetBlockNum() / tiles : 1;
    if (partitions > 1) {
      ProcessPartitionedRoutes(expertIds, groupOfRow, starts, groups, tiles, partitions);
      return;
    }
    // Zero all route slots first, so peer rows never retain stale graph data.
    auto out = output_.Get<half>();
    Duplicate(out, static_cast<half>(0), ELEMENTS);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    routed_ = false;
    for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
      for (int64_t row = 0; row < rows_; row += M) Store(out, tile, row, Min(M, rows_ - row));
    }
    routed_ = true;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for (uint32_t group = 0; group < groups; ++group) {
      const int64_t expert = expertIds[group];
      for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
        DataCopy(b1_.Get<int8_t>(), codes_[(expert * n_ + tile * N) * k_ / 2], N * k_ / 2);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        for (uint32_t row = starts[group]; row < starts[group + 1]; row += M) {
          Project(expert, tile, row, Min(M, starts[group + 1] - row));
        }
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
      }
    }
  }

  __aicore__ inline void ProcessPartitionedRoutes(const int32_t* expertIds, const uint32_t* groupOfRow,
                                                  const uint32_t* starts, uint32_t groups, uint32_t tiles,
                                                  uint32_t partitions) {
    // N=320 leaves four gate/up tiles on the eight 310P AI cores. Assign two
    // route partitions to each tile, halving activation gathers while keeping
    // every core occupied. Valid rows are always overwritten by their group
    // owner; only invalid/peer rows require explicit graph-replay clearing.
    const uint32_t tile = GetBlockIdx() % tiles;
    const uint32_t partition = GetBlockIdx() / tiles;
    auto out = output_.Get<half>();
    Duplicate(out, static_cast<half>(0), ELEMENTS);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    for (uint32_t row = partition; row < rows_; row += partitions) {
      if (groupOfRow[row] != MAX_ROUTES) continue;
      for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
        DataCopy(y_[static_cast<int64_t>(row) * n_ + tile * N + nb * BLOCK], out[nb * COLUMN_ELEMENTS], BLOCK);
      }
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for (uint32_t group = partition; group < groups; group += partitions) {
      const int64_t expert = expertIds[group];
      DataCopy(b1_.Get<int8_t>(), codes_[(expert * n_ + tile * N) * k_ / 2], N * k_ / 2);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      for (uint32_t row = starts[group]; row < starts[group + 1]; row += M) {
        Project(expert, tile, row, Min(M, starts[group + 1] - row));
      }
      SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
  }
  __aicore__ inline uint32_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
  __aicore__ inline void Store(LocalTensor<half> out, int64_t tile, int64_t row, uint32_t count) {
    DataCopyParams copy{static_cast<uint16_t>(count), 1, 0, static_cast<uint16_t>(n_ / BLOCK - 1)};
    for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
      if (routed_) {
        for (uint32_t m = 0; m < count; ++m) {
          DataCopy(y_[static_cast<int64_t>(routeRows_[row + m]) * n_ + tile * N + nb * BLOCK],
                   out[nb * COLUMN_ELEMENTS + m * BLOCK], BLOCK);
        }
      } else {
        DataCopy(y_[row * n_ + tile * N + nb * BLOCK], out[nb * COLUMN_ELEMENTS], copy);
      }
    }
  }

  __aicore__ inline void LoadWeight(int64_t group) {
    const uint32_t buffer = group % 2;
    LoadData2DParams load;
    // Consecutive output strips in L0B come from K-wide strips in L1.
    load.repeatTimes = N / BLOCK;
    load.srcStride = k_ / K0;
    load.ifTranspose = false;
    for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
      LoadData(b2_.Get<int8_t>()[buffer * B_BYTES + kb * N * K0 / 2].template ReinterpretCast<int4b_t>(),
               b1_.Get<int8_t>()[(group * GROUP + kb * K0) * BLOCK / 2].template ReinterpretCast<int4b_t>(), load);
    }
  }

  __aicore__ inline void LoadActivation(int64_t row, int64_t group, uint32_t count) {
    auto packed = packing_.Get<int8_t>();
    Duplicate(packed.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0), A_BYTES);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopyParams copy{static_cast<uint16_t>(count), 1, static_cast<uint16_t>(k_ / K0 - 1), 0};
    for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
      const int64_t src = row * k_ / 2 + group * GROUP / 2 + kb * K0 / 2;
      if (routed_) {
        for (uint32_t m = 0; m < count; ++m) {
          const int64_t routeSrc =
              static_cast<int64_t>(sourceRows_[routeRows_[row + m]]) * k_ / 2 + group * GROUP / 2 + kb * K0 / 2;
          DataCopy(packed[(kb * M + m) * K0 / 2], low_[routeSrc], K0 / 2);
          DataCopy(packed[A_BYTES + (kb * M + m) * K0 / 2], high_[routeSrc], K0 / 2);
        }
      } else {
        DataCopy(packed[kb * M * K0 / 2], low_[src], copy);
        DataCopy(packed[A_BYTES + kb * M * K0 / 2], high_[src], copy);
      }
    }
    DataCopyParams meta{static_cast<uint16_t>(count), 1, static_cast<uint16_t>(groups_ - 1), 0};
    auto xs = activation_.Get<float>();
    auto sums = xs[M * LANES];
    Duplicate(xs, 0.0f, 2 * M * LANES);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if (routed_) {
      for (uint32_t m = 0; m < count; ++m) {
        const int64_t index = (static_cast<int64_t>(sourceRows_[routeRows_[row + m]]) * groups_ + group) * LANES;
        DataCopy(xs[m * LANES], xs_[index], LANES);
        DataCopy(sums[m * LANES], sums_[index], LANES);
      }
    } else {
      DataCopy(xs, xs_[(row * groups_ + group) * LANES], meta);
      DataCopy(sums, sums_[(row * groups_ + group) * LANES], meta);
    }
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopy(a1_.Get<int8_t>(), packed, 2 * A_BYTES);
    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    LoadData2DParams load;
    load.repeatTimes = GROUP / K0;
    load.srcStride = M / BLOCK;
    load.ifTranspose = false;
    for (uint32_t limb = 0; limb < 2; ++limb) {
      for (uint32_t mb = 0; mb < M / BLOCK; ++mb) {
        LoadData(a2_.Get<int8_t>()[limb * A_BYTES + mb * BLOCK * GROUP / 2].template ReinterpretCast<int4b_t>(),
                 a1_.Get<int8_t>()[limb * A_BYTES + mb * BLOCK * K0 / 2].template ReinterpretCast<int4b_t>(), load);
      }
    }
  }

  __aicore__ inline void Product(uint32_t limb, uint32_t weightBuffer, LocalTensor<float> result,
                                 int64_t prefetchGroup = -1) {
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    MmadParams mm;
    mm.m = M;
    mm.n = N;
    mm.k = GROUP;
    mm.cmatrixInitVal = true;
    Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>()[limb * A_BYTES].template ReinterpretCast<int4b_t>(),
         b2_.Get<int8_t>()[weightBuffer * B_BYTES].template ReinterpretCast<int4b_t>(), mm);
    if (prefetchGroup >= 0) LoadWeight(prefetchGroup);
    SetFlag<HardEvent::M_V>(EVENT_ID0);
    WaitFlag<HardEvent::M_V>(EVENT_ID0);
    DataCopyParams copy{N / BLOCK, M / BLOCK, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(integers_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
    PipeBarrier<PIPE_V>();
    Cast(result, integers_.Get<int32_t>(), RoundMode::CAST_NONE, ELEMENTS);
    SetFlag<HardEvent::V_M>(EVENT_ID0);
    WaitFlag<HardEvent::V_M>(EVENT_ID0);
  }

  // Both INT4 activation limbs share the same packed weight tile. For
  // sparse rows, one M=2M Cube operation computes both integer products.
  __aicore__ inline void ProductPair(uint32_t weightBuffer, int64_t prefetchGroup = -1) {
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    MmadParams mm;
    mm.m = 2 * M;
    mm.n = N;
    mm.k = GROUP;
    mm.cmatrixInitVal = true;
    Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>().template ReinterpretCast<int4b_t>(),
         b2_.Get<int8_t>()[weightBuffer * B_BYTES].template ReinterpretCast<int4b_t>(), mm);
    if (prefetchGroup >= 0) LoadWeight(prefetchGroup);
    SetFlag<HardEvent::M_V>(EVENT_ID0);
    WaitFlag<HardEvent::M_V>(EVENT_ID0);
    DataCopyParams copy{N / BLOCK, 2 * M / BLOCK, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(integers_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
    PipeBarrier<PIPE_V>();
    Cast(result_.Get<float>(), integers_.Get<int32_t>(), RoundMode::CAST_NONE, 2 * ELEMENTS);
    SetFlag<HardEvent::V_M>(EVENT_ID0);
    WaitFlag<HardEvent::V_M>(EVENT_ID0);
  }

  // Sparse experts have fewer live rows than output strips. Traverse those
  // rows and vectorize across strips, leaving padded Cube rows unprocessed.
  template <uint32_t PRODUCT_ROWS>
  __aicore__ inline void CorrectRows(LocalTensor<float> low, LocalTensor<float> high, LocalTensor<float> accumulator,
                                     LocalTensor<float> sw, LocalTensor<float> zw, LocalTensor<float> ws,
                                     LocalTensor<float> xs, LocalTensor<float> sums, uint32_t count) {
    constexpr uint8_t PRODUCT_STRIDE = PRODUCT_ROWS * BLOCK / LANES;
    constexpr uint8_t ACCUMULATOR_STRIDE = M * BLOCK / LANES;
    constexpr uint8_t METADATA_STRIDE = BLOCK / LANES;
    constexpr uint8_t REPEATS = N / BLOCK;
    for (uint32_t row = 0; row < count; ++row)
      Muls(high[row * BLOCK], high[row * BLOCK], 16.0f, BLOCK, REPEATS, {1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Add(low[row * BLOCK], low[row * BLOCK], high[row * BLOCK], BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Mul(high[row * BLOCK], zw, sums[row * LANES], BLOCK, REPEATS, {1, 1, 0, PRODUCT_STRIDE, METADATA_STRIDE, 0});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Sub(low[row * BLOCK], low[row * BLOCK], high[row * BLOCK], BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, PRODUCT_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Add(low[row * BLOCK], low[row * BLOCK], ws, BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, METADATA_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Mul(low[row * BLOCK], low[row * BLOCK], sw, BLOCK, REPEATS,
          {1, 1, 1, PRODUCT_STRIDE, PRODUCT_STRIDE, METADATA_STRIDE});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Mul(low[row * BLOCK], low[row * BLOCK], xs[row * LANES], BLOCK, REPEATS,
          {1, 1, 0, PRODUCT_STRIDE, PRODUCT_STRIDE, 0});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < count; ++row)
      Add(accumulator[row * BLOCK], accumulator[row * BLOCK], low[row * BLOCK], BLOCK, REPEATS,
          {1, 1, 1, ACCUMULATOR_STRIDE, ACCUMULATOR_STRIDE, PRODUCT_STRIDE});
  }

  __aicore__ inline void Project(int64_t expert, int64_t tile, int64_t row, uint32_t count) {
    auto low = result_.Get<float>();
    auto high = low[ELEMENTS];
    auto accumulator = high[ELEMENTS];
    auto metadata = metadata_.Get<half>();
    auto sw = metadata_.Get<float>()[3 * N / 2];
    auto zw = sw[N];
    auto ws = zw[N];
    auto xs = activation_.Get<float>();
    auto sums = xs[M * LANES];
    Duplicate(accumulator, 0.0f, ELEMENTS);
    LoadWeight(0);
    for (int64_t group = 0; group < groups_; ++group) {
      // One strided DMA per metadata bank spans every N=16 strip in this
      // output tile. The existing packed layout and arithmetic stay unchanged.
      const int64_t index = ((expert * n_ / BLOCK + tile * N / BLOCK) * groups_ + group) * BLOCK;
      DataCopyParams metadataCopy{N / BLOCK, 1, static_cast<uint16_t>(groups_ - 1), 0};
      DataCopy(metadata, sw_[index], metadataCopy);
      DataCopy(metadata[N], zw_[index], metadataCopy);
      DataCopy(metadata[2 * N], ws_[index], metadataCopy);
      LoadActivation(row, group, count);
      // Alternate L0B buffers let next-group weight loads overlap integer GEMM.
      if constexpr (M <= 32) {
        if (count <= N / BLOCK) {
          ProductPair(group % 2, group + 1 < groups_ ? group + 1 : -1);
        } else {
          Product(0, group % 2, low, group + 1 < groups_ ? group + 1 : -1);
          Product(1, group % 2, high);
        }
      } else {
        Product(0, group % 2, low, group + 1 < groups_ ? group + 1 : -1);
        Product(1, group % 2, high);
      }
      SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
      WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(sw, metadata, RoundMode::CAST_NONE, 3 * N);
      PipeBarrier<PIPE_V>();
      Muls(ws, ws, 8.0f, N);
      if constexpr (M <= 32) {
        if (count <= N / BLOCK) {
          CorrectRows<2 * M>(low, low[M * BLOCK], accumulator, sw, zw, ws, xs, sums, count);
        } else {
          Muls(high, high, 16.0f, ELEMENTS);
          PipeBarrier<PIPE_V>();
          Add(low, low, high, ELEMENTS);
          PipeBarrier<PIPE_V>();
          // Each output strip is independent. Issue one correction stage across
          // all strips before synchronizing, instead of fencing every strip.
          for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
            Mul(high[nb * COLUMN_ELEMENTS], zw[nb * BLOCK], sums, BLOCK, count, {1, 1, 0, 2, 0, 1});
          }
          PipeBarrier<PIPE_V>();
          Sub(low, low, high, ELEMENTS);
          PipeBarrier<PIPE_V>();
          for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
            Add(low[nb * COLUMN_ELEMENTS], low[nb * COLUMN_ELEMENTS], ws[nb * BLOCK], BLOCK, count, {1, 1, 1, 2, 2, 0});
          }
          PipeBarrier<PIPE_V>();
          for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
            Mul(low[nb * COLUMN_ELEMENTS], low[nb * COLUMN_ELEMENTS], sw[nb * BLOCK], BLOCK, count, {1, 1, 1, 2, 2, 0});
          }
          PipeBarrier<PIPE_V>();
          for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
            Mul(low[nb * COLUMN_ELEMENTS], low[nb * COLUMN_ELEMENTS], xs, BLOCK, count, {1, 1, 0, 2, 2, 1});
          }
          PipeBarrier<PIPE_V>();
          Add(accumulator, accumulator, low, ELEMENTS);
        }
      } else {
        Muls(high, high, 16.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Add(low, low, high, ELEMENTS);
        PipeBarrier<PIPE_V>();
        // Each output strip is independent. Issue one correction stage across
        // all strips before synchronizing, instead of fencing every strip.
        for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
          Mul(high[nb * COLUMN_ELEMENTS], zw[nb * BLOCK], sums, BLOCK, count, {1, 1, 0, 2, 0, 1});
        }
        PipeBarrier<PIPE_V>();
        Sub(low, low, high, ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
          Add(low[nb * COLUMN_ELEMENTS], low[nb * COLUMN_ELEMENTS], ws[nb * BLOCK], BLOCK, count, {1, 1, 1, 2, 2, 0});
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
          Mul(low[nb * COLUMN_ELEMENTS], low[nb * COLUMN_ELEMENTS], sw[nb * BLOCK], BLOCK, count, {1, 1, 1, 2, 2, 0});
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t nb = 0; nb < N / BLOCK; ++nb) {
          Mul(low[nb * COLUMN_ELEMENTS], low[nb * COLUMN_ELEMENTS], xs, BLOCK, count, {1, 1, 0, 2, 2, 1});
        }
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, low, ELEMENTS);
      }
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
    }
    PipeBarrier<PIPE_V>();
    auto out = output_.Get<half>();
    Cast(out, accumulator, RoundMode::CAST_NONE, ELEMENTS);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(out, tile, row, count);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> packing_, integers_, result_, metadata_, activation_, output_, endCache_, routeCache_;
  GlobalTensor<int8_t> low_, high_, codes_;
  GlobalTensor<float> xs_, sums_;
  GlobalTensor<half> sw_, zw_, ws_, y_;
  GlobalTensor<int64_t> ends_;
  GlobalTensor<int32_t> routeIds_;
  uint32_t routeRows_[MAX_ROUTES];
  uint32_t sourceRows_[MAX_ROUTES];
  bool routed_;
  int64_t rows_, experts_, n_, k_, groups_, broadcastFactor_;
};
}  // namespace native_int4
#endif
