// SPDX-License-Identifier: Apache-2.0
// Native INT4 products on GLM's unchanged NZ W4 bank and 32x32 scale grid.
// Conservative first schedule: local nibble repacking, no FP16 weight GM tile.
#include "kernel_operator.h"
#ifndef GLM_INT4_OUTPUT_COLUMNS
  #define GLM_INT4_OUTPUT_COLUMNS 16
#endif

namespace {
using namespace AscendC;
constexpr uint32_t M = 16, N = GLM_INT4_OUTPUT_COLUMNS, K0 = 64, GROUP = 32, NZ_K = 256, NZ_N = 16;
static_assert(N == 16 || N == 32 || N == 64, "supported native output tiles are 16, 32, 64");
constexpr uint32_t PACKED_NZ = NZ_N * NZ_K / 2, FIELDS = N * NZ_K;
constexpr uint32_t A_BYTES = M * K0 / 2, B_BYTES = N * K0 / 2;
constexpr uint32_t ELEMENTS = M * N, GATHER_ELEMENTS = N * K0, LANES = 8;
class Projection {
 public:
  __aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR ends,
                             GM_ADDR y, const int64_t* config) {
    rows_ = config[0];
    experts_ = config[1];
    n_ = config[2];
    k_ = config[3];
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    codes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(codes));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    Allocate();
    PrepareOffsets();
    int64_t first = 0;
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const int64_t end = ends_.GetValue(expert);
      if (end < first || end > rows_) return;
      for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
        for (int64_t row = first; row < end; row += M) Project(expert, tile, row, end - row < M ? end - row : M);
        if (expert + 1 == experts_) {
          auto out = output_.Get<half>();
          Duplicate(out, static_cast<half>(0), ELEMENTS);
          PipeBarrier<PIPE_ALL>();
          for (int64_t row = end; row < rows_; row += M) Store(out, tile, row, rows_ - row < M ? rows_ - row : M);
          PipeBarrier<PIPE_ALL>();
        }
      }
      first = end;
    }
  }

 private:
  __aicore__ inline void Allocate() {
    pipe_.InitBuffer(a1_, 2 * A_BYTES);
    pipe_.InitBuffer(a2_, 2 * A_BYTES);
    pipe_.InitBuffer(b1_, B_BYTES);
    pipe_.InitBuffer(b2_, B_BYTES);
    pipe_.InitBuffer(c_, 2 * ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(raw_, PACKED_NZ);
    pipe_.InitBuffer(byteHalf_, PACKED_NZ * sizeof(half));
    pipe_.InitBuffer(quotient_, PACKED_NZ * sizeof(half));
    pipe_.InitBuffer(fields_, (FIELDS + N) * sizeof(half));
    pipe_.InitBuffer(integers_, PACKED_NZ * sizeof(int16_t));
    pipe_.InitBuffer(mask_, PACKED_NZ * sizeof(int16_t));
    pipe_.InitBuffer(sign_, PACKED_NZ * sizeof(int16_t));
    pipe_.InitBuffer(offsets_, (NZ_K / GROUP) * GATHER_ELEMENTS * sizeof(uint32_t));
    pipe_.InitBuffer(gathered_, GATHER_ELEMENTS * sizeof(half));
    pipe_.InitBuffer(packedB_, B_BYTES);
    pipe_.InitBuffer(weightFloat_, GATHER_ELEMENTS * sizeof(float));
    pipe_.InitBuffer(weightSum_, N * sizeof(float));
    pipe_.InitBuffer(packedA_, 2 * A_BYTES);
    pipe_.InitBuffer(products_, 2 * ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(productFloat_, 2 * ELEMENTS * sizeof(float));
    pipe_.InitBuffer(results_, 3 * ELEMENTS * sizeof(float));
    pipe_.InitBuffer(output_, ELEMENTS * sizeof(half));
  }
  __aicore__ inline void PrepareOffsets() {
    auto offsets = offsets_.Get<uint32_t>();
    for (uint32_t group = 0; group < NZ_K / GROUP; ++group)
      for (uint32_t channel = 0; channel < N; ++channel)
        for (uint32_t inner = 0; inner < K0; ++inner) {
          const uint32_t k = group * GROUP + inner;
          const uint32_t index = inner < GROUP ? (channel / NZ_N) * NZ_K * NZ_N + k * NZ_N + channel % NZ_N : FIELDS;
          offsets.SetValue(group * GATHER_ELEMENTS + channel * K0 + inner, index * sizeof(half));
        }
    Duplicate(mask_.Get<int16_t>(), static_cast<int16_t>(15), PACKED_NZ);
    Duplicate(sign_.Get<int16_t>(), static_cast<int16_t>(8), PACKED_NZ);
    Duplicate(fields_.Get<half>()[FIELDS], static_cast<half>(0), N);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Decode(int64_t expert, int64_t tile, int64_t kTile) {
    for (uint32_t strip = 0; strip < N / NZ_N; ++strip) {
      const int64_t offset = ((expert * (n_ / NZ_N) + tile * (N / NZ_N) + strip) * (k_ / NZ_K) + kTile) * PACKED_NZ;
      DataCopy(raw_.Get<uint8_t>(), codes_[offset], PACKED_NZ);
      PipeBarrier<PIPE_ALL>();
      auto bytes = byteHalf_.Get<half>();
      auto quotient = quotient_.Get<half>();
      auto integers = integers_.Get<int16_t>();
      auto fields = fields_.Get<half>()[strip * NZ_N * NZ_K];
      Cast(bytes, raw_.Get<uint8_t>(), RoundMode::CAST_NONE, PACKED_NZ);
      PipeBarrier<PIPE_V>();
      Muls(quotient, bytes, static_cast<half>(1.0f / 16.0f), PACKED_NZ);
      PipeBarrier<PIPE_V>();
      Adds(quotient, quotient, static_cast<half>(-15.0f / 32.0f), PACKED_NZ);
      PipeBarrier<PIPE_V>();
      Cast(integers, quotient, RoundMode::CAST_RINT, PACKED_NZ);
      PipeBarrier<PIPE_V>();
      Cast(quotient, integers, RoundMode::CAST_NONE, PACKED_NZ);
      PipeBarrier<PIPE_V>();
      Muls(fields, quotient, static_cast<half>(-16), PACKED_NZ);
      PipeBarrier<PIPE_V>();
      Add(fields, bytes, fields, PACKED_NZ);
      PipeBarrier<PIPE_V>();
      for (uint32_t field = 0; field < 2; ++field) {
        auto value = field == 0 ? fields : quotient;
        Cast(integers, value, RoundMode::CAST_RINT, PACKED_NZ);
        PipeBarrier<PIPE_V>();
        Add(integers, integers, sign_.Get<int16_t>(), PACKED_NZ);
        PipeBarrier<PIPE_V>();
        And(integers, integers, mask_.Get<int16_t>(), PACKED_NZ);
        PipeBarrier<PIPE_V>();
        Sub(integers, integers, sign_.Get<int16_t>(), PACKED_NZ);
        PipeBarrier<PIPE_V>();
        Cast(fields[field * PACKED_NZ], integers, RoundMode::CAST_NONE, PACKED_NZ);
        PipeBarrier<PIPE_V>();
      }
    }
  }
  __aicore__ inline void Weight(uint32_t group) {
    auto gathered = gathered_.Get<half>();
    Gather(gathered, fields_.Get<half>(), offsets_.Get<uint32_t>()[group * GATHER_ELEMENTS], static_cast<uint32_t>(0),
           GATHER_ELEMENTS);
    PipeBarrier<PIPE_V>();
    Cast(packedB_.Get<int8_t>().ReinterpretCast<int4b_t>(), gathered, RoundMode::CAST_NONE, GATHER_ELEMENTS);
    Cast(weightFloat_.Get<float>(), gathered, RoundMode::CAST_NONE, GATHER_ELEMENTS);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(weightSum_.Get<float>(), weightFloat_.Get<float>(), K0, N, 1, 1, K0 / LANES);
    PipeBarrier<PIPE_ALL>();
    DataCopy(b1_.Get<int8_t>(), packedB_.Get<int8_t>(), B_BYTES);
    PipeBarrier<PIPE_ALL>();
    LoadData2DParams load;
    load.repeatTimes = N / NZ_N;
    load.srcStride = 1;
    load.ifTranspose = false;
    LoadData(b2_.Get<int8_t>().ReinterpretCast<int4b_t>(), b1_.Get<int8_t>().ReinterpretCast<int4b_t>(), load);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Activation(int64_t row, int64_t group, uint32_t count) {
    auto packed = packedA_.Get<int8_t>();
    Duplicate(packed.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), A_BYTES);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t m = 0; m < count; ++m) {
      const int64_t offset = ((row + m) * (k_ / GROUP) + group) * (K0 / 2);
      DataCopy(packed[m * K0 / 2], low_[offset], K0 / 2);
      DataCopy(packed[A_BYTES + m * K0 / 2], high_[offset], K0 / 2);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(a1_.Get<int8_t>(), packed, 2 * A_BYTES);
    PipeBarrier<PIPE_ALL>();
    LoadData2DParams load;
    load.repeatTimes = 1;
    load.srcStride = 1;
    load.ifTranspose = false;
    for (uint32_t limb = 0; limb < 2; ++limb)
      LoadData(a2_.Get<int8_t>()[limb * A_BYTES].ReinterpretCast<int4b_t>(),
               a1_.Get<int8_t>()[limb * A_BYTES].ReinterpretCast<int4b_t>(), load);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Product() {
    MmadParams mm;
    mm.m = 2 * M;
    mm.n = N;
    mm.k = K0;
    mm.cmatrixInitVal = true;
    Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>().ReinterpretCast<int4b_t>(), b2_.Get<int8_t>().ReinterpretCast<int4b_t>(),
         mm);
    PipeBarrier<PIPE_ALL>();
    const DataCopyParams copy{N / NZ_N, 2, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(products_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
    PipeBarrier<PIPE_ALL>();
    auto raw = productFloat_.Get<float>();
    Cast(raw, products_.Get<int32_t>(), RoundMode::CAST_NONE, 2 * ELEMENTS);
    PipeBarrier<PIPE_V>();
    auto low = results_.Get<float>();
    auto high = low[ELEMENTS];
    // L0C is [N/16,2M,16]. Restore row-major low/high limb vectors without
    // changing the logical 32x32 scale groups.
    for (uint32_t strip = 0; strip < N / NZ_N; ++strip) {
      Adds(low[strip * NZ_N], raw[strip * 2 * M * NZ_N], 0.0f, NZ_N, M, {1, 1, N / LANES, NZ_N / LANES});
      Adds(high[strip * NZ_N], raw[strip * 2 * M * NZ_N + M * NZ_N], 0.0f, NZ_N, M, {1, 1, N / LANES, NZ_N / LANES});
    }
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Project(int64_t expert, int64_t tile, int64_t row, uint32_t count) {
    auto low = results_.Get<float>();
    auto high = low[ELEMENTS];
    auto accumulator = high[ELEMENTS];
    Duplicate(accumulator, 0.0f, ELEMENTS);
    PipeBarrier<PIPE_V>();
    for (int64_t kTile = 0; kTile < k_ / NZ_K; ++kTile) {
      Decode(expert, tile, kTile);
      for (uint32_t inner = 0; inner < NZ_K / GROUP; ++inner) {
        const int64_t group = kTile * (NZ_K / GROUP) + inner;
        Weight(inner);
        Activation(row, group, count);
        Product();
        for (uint32_t m = 0; m < count; ++m) {
          Muls(high[m * N], high[m * N], 16.0f, N);
          PipeBarrier<PIPE_V>();
          Add(low[m * N], low[m * N], high[m * N], N);
          PipeBarrier<PIPE_V>();
          Muls(high[m * N], weightSum_.Get<float>(), 8.0f, N);
          PipeBarrier<PIPE_V>();
          Add(low[m * N], low[m * N], high[m * N], N);
          PipeBarrier<PIPE_V>();
          const float sx = xs_.GetValue(((row + m) * (k_ / GROUP) + group) * LANES);
          for (uint32_t column = 0; column < N; column += GROUP) {
            const int64_t scaleIndex = (expert * (n_ / GROUP) + (tile * N + column) / GROUP) * (k_ / GROUP) + group;
            const float sw = static_cast<float>(static_cast<half>(scales_.GetValue(scaleIndex)));
            Muls(low[m * N + column], low[m * N + column], sw * sx, N < GROUP ? N : GROUP);
          }
          PipeBarrier<PIPE_V>();
          Add(accumulator[m * N], accumulator[m * N], low[m * N], N);
          PipeBarrier<PIPE_V>();
        }
      }
    }
    Cast(output_.Get<half>(), accumulator, RoundMode::CAST_NONE, ELEMENTS);
    PipeBarrier<PIPE_ALL>();
    Store(output_.Get<half>(), tile, row, count);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Store(LocalTensor<half> out, int64_t tile, int64_t row, uint32_t count) {
    for (uint32_t m = 0; m < count; ++m) DataCopy(y_[(row + m) * n_ + tile * N], out[m * N], N);
  }
  int64_t rows_, experts_, n_, k_;
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> raw_, byteHalf_, quotient_, fields_, integers_, mask_, sign_, offsets_;
  TBuf<TPosition::VECCALC> gathered_, packedB_, weightFloat_, weightSum_, packedA_, products_, results_, output_;
  TBuf<TPosition::VECCALC> productFloat_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<uint8_t> codes_;
  GlobalTensor<float> xs_, scales_;
  GlobalTensor<int64_t> ends_;
  GlobalTensor<half> y_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_w4a8_matmul_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                         GM_ADDR scales, GM_ADDR ends, GM_ADDR y, GM_ADDR tiling) {
  AscendC::InitSocState();
  int64_t config[4];
  const auto source = reinterpret_cast<__gm__ int64_t*>(tiling);
  for (unsigned i = 0; i < 4; ++i) config[i] = source[i];
  if (config[0] <= 0 || config[0] > 64 || config[1] <= 0 || config[2] <= 0 || config[2] % 32 != 0 || config[3] <= 0 ||
      config[3] % 256 != 0)
    return;
  Projection operation;
  operation.Run(low, high, xs, codes, scales, ends, y, config);
}
