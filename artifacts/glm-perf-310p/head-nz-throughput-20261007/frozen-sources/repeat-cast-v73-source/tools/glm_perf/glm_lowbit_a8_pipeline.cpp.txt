// SPDX-License-Identifier: Apache-2.0
// Native INT4 products on unchanged GLM NZ W2/W3/W4 banks and block32 scales.
// Full tile repacking with exact nibble normalization and Cube-computed weight sums.
#include "kernel_operator.h"
#ifndef GLM_INT4_OUTPUT_COLUMNS
  #define GLM_INT4_OUTPUT_COLUMNS 128
#endif

namespace {
using namespace AscendC;
constexpr uint32_t M = 16, N = GLM_INT4_OUTPUT_COLUMNS, K0 = 64, GROUP = 32, NZ_K = 256, NZ_N = 16;
static_assert(N == 16 || N == 32 || N == 64 || N == 128, "supported native output tiles are 16..128");
constexpr uint32_t PACKED_NZ = NZ_N * NZ_K / 2, RAW_BYTES = N * NZ_K / 2;
constexpr uint32_t GROUPS = NZ_K / GROUP, WORDS_PER_ROW = K0 / 4;
constexpr uint32_t PREPARED_WORDS = GROUPS * N * WORDS_PER_ROW, MAX_ROWS = M - 1;
constexpr uint32_t A_BYTES = M * K0 / 2, B_BYTES = N * K0 / 2;
constexpr uint32_t ELEMENTS = M * N, GATHER_ELEMENTS = N * K0, LANES = 8;
class Projection {
 public:
  __aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR ends,
                             GM_ADDR y, GM_ADDR metadata, const int64_t* config) {
    rows_ = config[0];
    experts_ = config[1];
    n_ = config[2];
    k_ = config[3];
    bits_ = config[4];
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    codes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(codes));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    metadata_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
    Allocate();
    PrepareOffsets();
    int64_t first = 0;
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const int64_t end = ends_.GetValue(expert);
      if (end < first || end > rows_) return;
      for (int64_t tile = GetBlockIdx(); tile < n_ / N; tile += GetBlockNum()) {
        for (int64_t row = first; row < end; row += MAX_ROWS)
          Project(expert, tile, row, end - row < MAX_ROWS ? end - row : MAX_ROWS);
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
    pipe_.InitBuffer(raw_, RAW_BYTES + 64);
    pipe_.InitBuffer(offsets_, PREPARED_WORDS * sizeof(uint32_t));
    pipe_.InitBuffer(packedB_, PREPARED_WORDS * sizeof(uint16_t));
    pipe_.InitBuffer(gathered_, PREPARED_WORDS * sizeof(uint16_t));
    pipe_.InitBuffer(mask_, PREPARED_WORDS * sizeof(uint16_t));
    pipe_.InitBuffer(outputOffsets_, N * sizeof(uint32_t));
    pipe_.InitBuffer(outputRow_, N * sizeof(half));
    pipe_.InitBuffer(packedA_, 2 * A_BYTES);
    pipe_.InitBuffer(products_, 2 * ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(productFloat_, PREPARED_WORDS * sizeof(half));
    pipe_.InitBuffer(results_, 3 * ELEMENTS * sizeof(float));
    pipe_.InitBuffer(output_, ELEMENTS * sizeof(half));
  }
  __aicore__ inline void PrepareOffsets() {
    auto offsets = offsets_.Get<uint32_t>();
    DataCopy(offsets, metadata_, PREPARED_WORDS);
    auto outputOffsets = outputOffsets_.Get<uint32_t>();
    for (uint32_t channel = 0; channel < N; ++channel) {
      const uint32_t physical = channel / 2 + (channel % 2 ? N / 2 : 0);
      outputOffsets.SetValue(channel, physical * sizeof(half));
    }
    const uint32_t rawBytes = N * NZ_K * bits_ / 8;
    Duplicate(raw_.Get<uint16_t>()[rawBytes / 2], static_cast<uint16_t>(0), bits_ == 3 ? 288 : 32);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Normalize(uint32_t field, bool upper) {
    auto words = gathered_.Get<uint16_t>();
    auto factors = productFloat_.Get<half>();
    const uint32_t halfWords = (N / 2) * WORDS_PER_ROW;
    for (uint32_t group = 0; group < GROUPS; ++group) {
      const uint32_t shift = bits_ == 3 ? (3 * group) % 8 : bits_ * (group / bits_);
      const uint32_t lowBits = bits_ < 8 - shift ? bits_ : 8 - shift;
      const uint32_t count = upper ? bits_ - lowBits : lowBits;
      for (uint32_t odd = 0; odd < 2; ++odd) {
        const uint32_t start = group * N * WORDS_PER_ROW + odd * halfWords;
        const uint32_t phase = (upper ? 0 : shift) + 8 * odd;
        Duplicate(mask_.Get<uint16_t>()[start], static_cast<uint16_t>(((1 << count) - 1) << phase), halfWords);
        Duplicate(factors[start], static_cast<half>(1.0f / (1 << phase)), halfWords);
      }
    }
    PipeBarrier<PIPE_V>();
    And(words, words, mask_.Get<uint16_t>(), PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    auto normalized = words.ReinterpretCast<half>();
    Cast(normalized, words.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    Mul(normalized, normalized, factors, PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    Cast(words.ReinterpretCast<int16_t>(), normalized, RoundMode::CAST_RINT, PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    for (uint32_t group = 0; group < GROUPS; ++group) {
      const uint32_t shift = bits_ == 3 ? (3 * group) % 8 : bits_ * (group / bits_);
      const uint32_t lowBits = bits_ < 8 - shift ? bits_ : 8 - shift;
      const uint32_t count = upper ? bits_ - lowBits : lowBits;
      Duplicate(factors.ReinterpretCast<uint16_t>()[group * N * WORDS_PER_ROW], static_cast<uint16_t>((1 << count) - 1),
                N * WORDS_PER_ROW);
    }
    PipeBarrier<PIPE_V>();
    And(words, words, factors.ReinterpretCast<uint16_t>(), PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    for (uint32_t group = 0; group < GROUPS; ++group) {
      const uint32_t shift = bits_ == 3 ? (3 * group) % 8 : bits_ * (group / bits_);
      const uint32_t lowBits = bits_ < 8 - shift ? bits_ : 8 - shift;
      Duplicate(factors.ReinterpretCast<int16_t>()[group * N * WORDS_PER_ROW],
                static_cast<int16_t>(1 << (4 * field + (upper ? lowBits : 0))), N * WORDS_PER_ROW);
    }
    PipeBarrier<PIPE_V>();
    Mul(words.ReinterpretCast<int16_t>(), words.ReinterpretCast<int16_t>(), factors.ReinterpretCast<int16_t>(),
        PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    Or(packedB_.Get<uint16_t>(), packedB_.Get<uint16_t>(), words, PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Decode(int64_t expert, int64_t tile, int64_t kTile) {
    const uint32_t packedBytes = NZ_N * NZ_K * bits_ / 8;
    for (uint32_t strip = 0; strip < N / NZ_N; ++strip) {
      const int64_t offset = ((expert * (n_ / NZ_N) + tile * (N / NZ_N) + strip) * (k_ / NZ_K) + kTile) * packedBytes;
      DataCopy(raw_.Get<uint8_t>()[strip * packedBytes], codes_[offset], packedBytes);
    }
    PipeBarrier<PIPE_ALL>();
    auto packed = packedB_.Get<uint16_t>();
    auto words = gathered_.Get<uint16_t>();
    Duplicate(packed, static_cast<uint16_t>(0), PREPARED_WORDS);
    PipeBarrier<PIPE_V>();
    for (uint32_t field = 0; field < 4; ++field) {
      Gather(words, raw_.Get<uint16_t>(), offsets_.Get<uint32_t>(), field * NZ_N, PREPARED_WORDS);
      PipeBarrier<PIPE_V>();
      Normalize(field, false);
      if (bits_ == 3) {
        // W3 fields 2 and 5 cross byte planes. Other groups use a zero mask.
        Gather(words, raw_.Get<uint16_t>(), offsets_.Get<uint32_t>(), field * NZ_N + NZ_N * NZ_K / 8, PREPARED_WORDS);
        PipeBarrier<PIPE_V>();
        Normalize(field, true);
      }
    }
    if (bits_ < 4) {
      // Extend the signed W2/W3 fields into signed INT4 lanes without changing
      // their values: W3 sign bit 4 maps to 12, W2 sign bit 2 maps to 14.
      const uint16_t signMask = bits_ == 3 ? 0x4444 : 0x2222;
      const int16_t signFactor = bits_ == 3 ? 2 : 6;
      Duplicate(mask_.Get<uint16_t>(), signMask, PREPARED_WORDS);
      PipeBarrier<PIPE_V>();
      And(words, packed, mask_.Get<uint16_t>(), PREPARED_WORDS);
      PipeBarrier<PIPE_V>();
      Muls(words.ReinterpretCast<int16_t>(), words.ReinterpretCast<int16_t>(), signFactor, PREPARED_WORDS);
      PipeBarrier<PIPE_V>();
      Or(packed, packed, words, PREPARED_WORDS);
    }
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void Weight(uint32_t group) {
    DataCopy(b1_.Get<int8_t>(), packedB_.Get<int8_t>()[group * B_BYTES], B_BYTES);
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
    // Each routed token occurs once per expert. Small decode groups leave
    // row 15 unused; compute sum(weights) in that Cube row for bias correction.
    Duplicate(packed.ReinterpretCast<uint16_t>()[MAX_ROWS * K0 / 4], static_cast<uint16_t>(0x1111), K0 / 4);
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
          Muls(high[m * N], low[MAX_ROWS * N], 8.0f, N);
          PipeBarrier<PIPE_V>();
          Add(low[m * N], low[m * N], high[m * N], N);
          PipeBarrier<PIPE_V>();
          const float sx = xs_.GetValue(((row + m) * (k_ / GROUP) + group) * LANES);
          for (uint32_t column = 0; column < N; column += NZ_N) {
            const uint32_t logical = column < N / 2 ? 2 * column : 2 * (column - N / 2) + 1;
            const int64_t scaleIndex = (expert * (n_ / GROUP) + (tile * N + logical) / GROUP) * (k_ / GROUP) + group;
            const float sw = static_cast<float>(static_cast<half>(scales_.GetValue(scaleIndex)));
            Muls(low[m * N + column], low[m * N + column], sw * sx, NZ_N);
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
    for (uint32_t m = 0; m < count; ++m) {
      Gather(outputRow_.Get<half>(), out[m * N], outputOffsets_.Get<uint32_t>(), static_cast<uint32_t>(0), N);
      PipeBarrier<PIPE_ALL>();
      DataCopy(y_[(row + m) * n_ + tile * N], outputRow_.Get<half>(), N);
      PipeBarrier<PIPE_ALL>();
    }
  }
  int64_t rows_, experts_, n_, k_, bits_;
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> raw_, mask_, offsets_, outputOffsets_, outputRow_;
  TBuf<TPosition::VECCALC> gathered_, packedB_, packedA_, products_, results_, output_;
  TBuf<TPosition::VECCALC> productFloat_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<uint8_t> codes_;
  GlobalTensor<float> xs_, scales_;
  GlobalTensor<int64_t> ends_;
  GlobalTensor<half> y_;
  GlobalTensor<uint32_t> metadata_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_w4a8_matmul_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                         GM_ADDR scales, GM_ADDR ends, GM_ADDR y, GM_ADDR tiling) {
  AscendC::InitSocState();
  int64_t config[5];
  const auto source = reinterpret_cast<__gm__ int64_t*>(tiling);
  for (unsigned i = 0; i < 5; ++i) config[i] = source[i];
  if (config[0] <= 0 || config[0] > 65536 || config[1] <= 0 || config[2] <= 0 || config[2] % 32 != 0 ||
      config[3] <= 0 || config[3] % 256 != 0 || config[4] < 2 || config[4] > 4)
    return;
  Projection operation;
  operation.Run(low, high, xs, codes, scales, ends, y, tiling + 8 * sizeof(int64_t), config);
}
