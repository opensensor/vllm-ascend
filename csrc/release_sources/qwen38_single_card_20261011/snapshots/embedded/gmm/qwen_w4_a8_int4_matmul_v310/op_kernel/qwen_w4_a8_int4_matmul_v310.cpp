// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"

// EXPERIMENTAL: two native mad_s4 instructions per activation/weight group.
// Packed signed W4 is loaded directly into L0B, never unpacked into FP16.
// The FP16 metadata/output epilogue does not perform a floating-point GEMM.
namespace {
using namespace AscendC;
constexpr uint32_t TILE = 16, GROUP = 128, FRACTAL_K = 64;
constexpr uint32_t PACKED_TILE = TILE * GROUP / 2, ELEMENTS = TILE * TILE;
constexpr int64_t DECODE_ROUTE_LIMIT = 128;
constexpr int64_t MODEL_C1_ROUTE_LIMIT = 30;
constexpr int64_t MODEL_GATE_UP_OUTPUTS = 1280;
constexpr int64_t MODEL_GATE_UP_INPUTS = 2560;
constexpr int64_t MODEL_DOWN_OUTPUTS = 2560;
constexpr int64_t MODEL_DOWN_INPUTS = 640;
constexpr uint32_t MODEL_DECODE_COLUMNS = 320;

class NativeInt4 {
 public:
  __aicore__ inline void Init(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                              GM_ADDR offset, GM_ADDR weight_sum, GM_ADDR ends, GM_ADDR y,
                              __gm__ const QwenW4A8Int4KernelTilingData* td) {
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
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
    offset_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(offset));
    ws_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight_sum));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    pipe_.InitBuffer(a1_, PACKED_TILE);
    pipe_.InitBuffer(b1_, PACKED_TILE);
    pipe_.InitBuffer(a2_, PACKED_TILE);
    pipe_.InitBuffer(b2_, PACKED_TILE);
    pipe_.InitBuffer(c_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(ub_, 16384);
  }

  __aicore__ inline void Process() {
    const int64_t tiles = n_ / TILE;
    // Read each group's boundaries once per core, not once per output tile.
    // Sparse decode has mostly empty experts; avoid redundant GM scalar loads
    // from flattening the expert/N product into the outer loop.
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const int64_t start = expert == 0 ? 0 : ends_.GetValue(expert - 1);
      const int64_t end = ends_.GetValue(expert);
      if (start < 0 || end < start || end > rows_) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid W4A8 group boundaries"); });
        return;
      }
      if (start == end && expert + 1 != experts_) continue;
      for (int64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
        if (expert + 1 == experts_) {
          auto zero = ub_.Get<half>();
          Duplicate(zero, static_cast<half>(0), ELEMENTS);
          SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
          for (int64_t row = end; row < rows_; row += TILE) {
            Store(zero, row, tile, Min(rows_ - row, static_cast<int64_t>(TILE)));
          }
          SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for (int64_t row = start; row < end; row += TILE) {
          Project(expert, tile, row, Min(end - row, static_cast<int64_t>(TILE)));
        }
      }
    }
  }

 private:
  __aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
  __aicore__ inline void Store(LocalTensor<half> out, int64_t row, int64_t tile, uint32_t count) {
    DataCopyParams copy;
    copy.blockCount = count;
    copy.blockLen = 1;
    copy.srcStride = 0;
    copy.dstStride = n_ / TILE - 1;
    DataCopy(y_[row * n_ + tile * TILE], out, copy);
  }
  __aicore__ inline void Product(GlobalTensor<int8_t> source, int64_t row, int64_t group, uint32_t count,
                                 LocalTensor<float> result) {
    auto bytes = ub_.Get<int8_t>();
    Duplicate(bytes.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), PACKED_TILE / 2);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopyParams copy;
    copy.blockCount = count;
    copy.blockLen = 1;
    copy.srcStride = k_ / FRACTAL_K - 1;
    copy.dstStride = 0;
    for (uint32_t kb = 0; kb < GROUP / FRACTAL_K; ++kb) {
      DataCopy(bytes[kb * TILE * FRACTAL_K / 2], source[row * k_ / 2 + group * GROUP / 2 + kb * FRACTAL_K / 2], copy);
    }
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopy(a1_.Get<int8_t>(), bytes, PACKED_TILE);
    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    LoadData2DParams load;
    load.repeatTimes = GROUP / FRACTAL_K;
    load.srcStride = 1;
    load.ifTranspose = false;
    LoadData(a2_.Get<int8_t>().ReinterpretCast<int4b_t>(), a1_.Get<int8_t>().ReinterpretCast<int4b_t>(), load);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    MmadParams mm;
    mm.m = TILE;
    mm.n = TILE;
    mm.k = GROUP;
    mm.cmatrixInitVal = true;
    Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>().ReinterpretCast<int4b_t>(), b2_.Get<int8_t>().ReinterpretCast<int4b_t>(),
         mm);
    SetFlag<HardEvent::M_V>(EVENT_ID0);
    WaitFlag<HardEvent::M_V>(EVENT_ID0);
    auto integers = ub_.Get<int32_t>()[PACKED_TILE / sizeof(int32_t)];
    DataCopyParams fromCube;
    fromCube.blockCount = 1;
    fromCube.blockLen = 1;
    fromCube.srcStride = 0;
    fromCube.dstStride = 0;
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(integers, c_.Get<int32_t>(), fromCube, enhanced);
    Cast(result, integers, RoundMode::CAST_NONE, ELEMENTS);
    SetFlag<HardEvent::V_M>(EVENT_ID0);
    WaitFlag<HardEvent::V_M>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  }
  __aicore__ inline void Project(int64_t expert, int64_t tile, int64_t row, uint32_t count) {
    // UB regions are disjoint: packing 0:1024, integer C 1024:2048,
    // low/high/accumulator 2048:5120, metadata/temporary 5120 onwards.
    auto low = ub_.Get<float>()[512];
    auto high = ub_.Get<float>()[768];
    auto accumulator = ub_.Get<float>()[1024];
    auto sw = ub_.Get<float>()[1280];
    auto zw = ub_.Get<float>()[1296];
    auto sumw = ub_.Get<float>()[1312];
    auto temporary = ub_.Get<float>()[1328];
    auto metadata = ub_.Get<half>()[2816];
    Duplicate(accumulator, 0.0f, ELEMENTS);
    for (int64_t group = 0; group < groups_; ++group) {
      const int64_t index = ((expert * (n_ / TILE) + tile) * groups_ + group);
      DataCopy(b1_.Get<int8_t>(), codes_[index * PACKED_TILE], PACKED_TILE);
      DataCopy(metadata, scale_[index * TILE], TILE);
      DataCopy(metadata[TILE], offset_[index * TILE], TILE);
      DataCopy(metadata[2 * TILE], ws_[index * TILE], TILE);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      LoadData2DParams load;
      load.repeatTimes = GROUP / FRACTAL_K;
      load.srcStride = 1;
      load.ifTranspose = false;
      LoadData(b2_.Get<int8_t>().ReinterpretCast<int4b_t>(), b1_.Get<int8_t>().ReinterpretCast<int4b_t>(), load);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(sw, metadata, RoundMode::CAST_NONE, TILE);
      Cast(zw, metadata[TILE], RoundMode::CAST_NONE, TILE);
      Cast(sumw, metadata[2 * TILE], RoundMode::CAST_NONE, TILE);
      Muls(sumw, sumw, 8.0f, TILE);
      Product(low_, row, group, count, low);
      Product(high_, row, group, count, high);
      Muls(high, high, 16.0f, ELEMENTS);
      Add(low, low, high, ELEMENTS);
      for (uint32_t m = 0; m < count; ++m) {
        const float sum = sums_.GetValue((row + m) * groups_ + group);
        const float scale = xs_.GetValue((row + m) * groups_ + group);
        Muls(temporary, zw, -sum, TILE);
        Add(temporary, temporary, sumw, TILE);
        Add(temporary, temporary, low[m * TILE], TILE);
        Mul(temporary, temporary, sw, TILE);
        Muls(temporary, temporary, scale, TILE);
        Add(accumulator[m * TILE], accumulator[m * TILE], temporary, TILE);
      }
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    auto out = ub_.Get<half>();
    Cast(out, accumulator, RoundMode::CAST_NONE, ELEMENTS);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(out, row, tile, count);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> ub_;
  GlobalTensor<int8_t> low_, high_, codes_;
  GlobalTensor<float> xs_, sums_;
  GlobalTensor<half> scale_, offset_, ws_, y_;
  GlobalTensor<int64_t> ends_;
  int64_t rows_, experts_, n_, k_, groups_;
};
template <uint32_t N>
__aicore__ inline void RunSchedule(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                                   GM_ADDR offset, GM_ADDR weight_sum, GM_ADDR ends, GM_ADDR y,
                                   __gm__ const QwenW4A8Int4KernelTilingData* td) {
  if (td->numRows <= DECODE_ROUTE_LIMIT) {
    if (td->numRows <= MODEL_C1_ROUTE_LIMIT) {
      native_int4::Schedule<16, N, false, N == MODEL_DECODE_COLUMNS> op;
      op.Init(low, high, xs, sums, codes, scale, offset, weight_sum, ends, nullptr, y, td);
      op.Process();
    } else {
      native_int4::Schedule<16, N> op;
      op.Init(low, high, xs, sums, codes, scale, offset, weight_sum, ends, nullptr, y, td);
      op.Process();
    }
  } else if (td->numRows > td->numExperts * 64) {
    native_int4::Schedule<128> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weight_sum, ends, nullptr, y, td);
    op.Process();
  } else {
    native_int4::Schedule<32, N> op;
    op.Init(low, high, xs, sums, codes, scale, offset, weight_sum, ends, nullptr, y, td);
    op.Process();
  }
}
}  // namespace

extern "C" __global__ __aicore__ void qwen_w4_a8_int4_matmul_v310(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                                  GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                                  GM_ADDR weight_sum, GM_ADDR ends, GM_ADDR y,
                                                                  GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(tiling);
  if (td->metadataLanes == 8) {
    // Keep every core equally occupied at the model's projection widths.
    // Dense prefill retains N=64 to fit the M=128 accumulator in UB.
    constexpr uint32_t WIDE_COLUMNS = 160, MEDIUM_COLUMNS = 80;
    const bool modelC1GateUp = td->numRows <= MODEL_C1_ROUTE_LIMIT && td->nDim == MODEL_GATE_UP_OUTPUTS &&
                               td->kDim == MODEL_GATE_UP_INPUTS;
    const bool modelDecodeDown = td->numRows <= DECODE_ROUTE_LIMIT && td->nDim == MODEL_DOWN_OUTPUTS &&
                                 td->kDim == MODEL_DOWN_INPUTS;
    if (modelC1GateUp || modelDecodeDown) {
      // One 320-column tile per AI core halves route-plan and activation
      // preparation for the model's down projection. Gate/up has only four
      // tiles at this width, so its c1 path splits route groups across two
      // cores per tile. Larger gate/up concurrency and every prefill shape
      // keep their proven schedules.
      RunSchedule<MODEL_DECODE_COLUMNS>(low, high, xs, sums, codes, scale, offset, weight_sum, ends, y, td);
    } else if (td->nDim % (GetBlockNum() * WIDE_COLUMNS) == 0) {
      RunSchedule<WIDE_COLUMNS>(low, high, xs, sums, codes, scale, offset, weight_sum, ends, y, td);
    } else if (td->nDim % (GetBlockNum() * MEDIUM_COLUMNS) == 0) {
      RunSchedule<MEDIUM_COLUMNS>(low, high, xs, sums, codes, scale, offset, weight_sum, ends, y, td);
    } else {
      RunSchedule<64>(low, high, xs, sums, codes, scale, offset, weight_sum, ends, y, td);
    }
    return;
  }
  NativeInt4 op;
  op.Init(low, high, xs, sums, codes, scale, offset, weight_sum, ends, y, td);
  op.Process();
}
