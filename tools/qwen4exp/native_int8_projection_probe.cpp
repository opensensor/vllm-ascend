// SPDX-License-Identifier: Apache-2.0
// Experimental single-expert G128 projection; raw INT8 or packed W4 weights.
// All groups accumulate in FP32, with the same sw-then-sx multiplication order.
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t N = 160, GROUP = 128, BLOCK = 16, K0 = 32, LANES = 8, MAX_K = 2560;
template <bool EXPAND_PACKED_WEIGHTS = false, uint32_t ROW_TILE = 16, bool LOOKUP_DECODE = false>
class Int8ProjectionProbe {
  static constexpr uint32_t M = ROW_TILE;
  static constexpr uint32_t ELEMENTS = M * N;
  static_assert(M == BLOCK || M == 2 * BLOCK, "supported INT8 row tile is 16 or 32");

 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR inputScale, GM_ADDR output,
                             GM_ADDR descriptor, GM_ADDR offset = nullptr, GM_ADDR interleave = nullptr,
                             GM_ADDR lookup = nullptr) {
    auto config = reinterpret_cast<__gm__ int64_t*>(descriptor);
    const int64_t rows = config[0], k = config[1], n = config[2], groups = k / GROUP;
    ASCENDC_ASSERT(rows > 0 && k > 0 && k <= MAX_K && k % GROUP == 0 && n > 0 && n % N == 0,
                   { KERNEL_LOG(KERNEL_ERROR, "invalid INT8 projection probe geometry"); });
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(weight));
    sw_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
    sx_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputScale));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    if constexpr (EXPAND_PACKED_WEIGHTS) {
      zw_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(offset));
      indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(interleave));
      if constexpr (LOOKUP_DECODE) {
        lookup_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(lookup));
        pipe_.InitBuffer(decoder_, LOOKUP_DECODE_BYTES);
      } else
        pipe_.InitBuffer(decoder_, DECODE_BYTES);
    }
    pipe_.InitBuffer(a1_, M * GROUP);
    pipe_.InitBuffer(b1_, N * MAX_K);
    pipe_.InitBuffer(a2_, M * GROUP);
    pipe_.InitBuffer(b2_, N * GROUP);
    pipe_.InitBuffer(c_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(packed_, M * GROUP);
    pipe_.InitBuffer(integer_, ELEMENTS * sizeof(int32_t));
    pipe_.InitBuffer(product_, 2 * ELEMENTS * sizeof(float));
    pipe_.InitBuffer(metadata_, N * sizeof(half) + N * sizeof(float) + M * LANES * sizeof(float));
    pipe_.InitBuffer(result_, ELEMENTS * sizeof(half));
    auto packed = packed_.Get<int8_t>();
    auto product = product_.Get<float>();
    auto accumulator = product[ELEMENTS];
    auto scales = metadata_.Get<half>();
    auto weightScale = metadata_.Get<float>()[N / 2];
    auto activationScale = weightScale[N];
    for (int64_t tile = GetBlockIdx(); tile < n / N; tile += GetBlockNum()) {
      // One resident full-K tile; no per-group reads of the weight bank from GM.
      if constexpr (LOOKUP_DECODE)
        CacheLookupWeights(tile, groups);
      else if constexpr (EXPAND_PACKED_WEIGHTS)
        CachePackedWeights(tile, k, groups);
      else
        DataCopy(b1_.Get<int8_t>(), w_[tile * N * k], N * k);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
      for (int64_t first = 0; first < rows; first += M) {
        const uint32_t count = rows - first < M ? rows - first : M;
        Duplicate(accumulator, 0.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (int64_t group = 0; group < groups; ++group) {
          Duplicate(packed.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), M * GROUP / 2);
          Duplicate(activationScale, 0.0f, M * LANES);
          SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
          DataCopyParams inputCopy{static_cast<uint16_t>(count), 1, static_cast<uint16_t>(k / K0 - 1), 0};
          for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
            DataCopy(packed[kb * M * K0], x_[first * k + group * GROUP + kb * K0], inputCopy);
          }
          DataCopyParams scaleCopy{N / BLOCK, 1, static_cast<uint16_t>(groups - 1), 0};
          DataCopy(scales, sw_[(tile * N / BLOCK * groups + group) * BLOCK], scaleCopy);
          DataCopyParams activationCopy{static_cast<uint16_t>(count), 1, static_cast<uint16_t>(groups - 1), 0};
          DataCopy(activationScale, sx_[(first * groups + group) * LANES], activationCopy);
          SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
          DataCopy(a1_.Get<int8_t>(), packed, M * GROUP);
          SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
          LoadData2DParams loadA;
          loadA.repeatTimes = GROUP / K0;
          loadA.srcStride = M / BLOCK;
          loadA.ifTranspose = false;
          for (uint32_t mb = 0; mb < M / BLOCK; ++mb)
            LoadData(a2_.Get<int8_t>()[mb * BLOCK * GROUP], a1_.Get<int8_t>()[mb * BLOCK * K0], loadA);
          LoadData2DParams loadB;
          loadB.repeatTimes = N / BLOCK;
          loadB.srcStride = k / K0;
          loadB.ifTranspose = false;
          for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
            LoadData(b2_.Get<int8_t>()[kb * N * K0], b1_.Get<int8_t>()[(group * GROUP + kb * K0) * BLOCK], loadB);
          }
          SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
          WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
          MmadParams mm;
          mm.m = M;
          mm.n = N;
          mm.k = GROUP;
          mm.cmatrixInitVal = true;
          Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>(), b2_.Get<int8_t>(), mm);
          SetFlag<HardEvent::M_V>(EVENT_ID0);
          WaitFlag<HardEvent::M_V>(EVENT_ID0);
          DataCopyParams copy{N / BLOCK, M / BLOCK, 0, 0};
          DataCopyEnhancedParams enhanced;
          enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
          DataCopy(integer_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
          PipeBarrier<PIPE_V>();
          Cast(product, integer_.Get<int32_t>(), RoundMode::CAST_NONE, ELEMENTS);
          Cast(weightScale, scales, RoundMode::CAST_NONE, N);
          PipeBarrier<PIPE_V>();
          // Visit the shorter dimension in scalar control. Full row tiles
          // broadcast one weight scale across rows in a vector instruction.
          if (count > N / BLOCK) {
            for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
              Mul(product[nb * M * BLOCK], product[nb * M * BLOCK], weightScale[nb * BLOCK], BLOCK, count,
                  {1, 1, 1, BLOCK / LANES, BLOCK / LANES, 0});
          } else {
            for (uint32_t row = 0; row < count; ++row)
              Mul(product[row * BLOCK], product[row * BLOCK], weightScale, BLOCK, N / BLOCK,
                  {1, 1, 1, M * BLOCK / LANES, M * BLOCK / LANES, BLOCK / LANES});
          }
          PipeBarrier<PIPE_V>();
          if (count > N / BLOCK) {
            for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
              Mul(product[nb * M * BLOCK], product[nb * M * BLOCK], activationScale, BLOCK, count,
                  {1, 1, 0, BLOCK / LANES, BLOCK / LANES, 1});
          } else {
            for (uint32_t row = 0; row < count; ++row)
              Mul(product[row * BLOCK], product[row * BLOCK], activationScale[row * LANES], BLOCK, N / BLOCK,
                  {1, 1, 0, M * BLOCK / LANES, M * BLOCK / LANES, 0});
          }
          PipeBarrier<PIPE_V>();
          Add(accumulator, accumulator, product, ELEMENTS);
          PipeBarrier<PIPE_V>();
          SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
          WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
          SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        Cast(result_.Get<half>(), accumulator, RoundMode::CAST_NONE, ELEMENTS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams store{static_cast<uint16_t>(count), 1, 0, static_cast<uint16_t>(n / BLOCK - 1)};
        for (uint32_t nb = 0; nb < N / BLOCK; ++nb)
          DataCopy(y_[first * n + tile * N + nb * BLOCK], result_.Get<half>()[nb * M * BLOCK], store);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      }
      SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
    pipe_.Destroy();
  }

 private:
  static constexpr uint32_t DECODE_COLUMNS = 80;
  static constexpr uint32_t RAW_ELEMENTS = DECODE_COLUMNS * GROUP / 2;
  static constexpr uint32_t DECODE_ELEMENTS = DECODE_COLUMNS * GROUP;
  static constexpr uint32_t DECODE_BASE_BYTES = 6 * RAW_ELEMENTS + 13 * DECODE_ELEMENTS;
  static constexpr uint32_t DECODE_BYTES =
      DECODE_BASE_BYTES + DECODE_COLUMNS * (sizeof(half) + (1 + LANES) * sizeof(float));
  static constexpr uint32_t LOOKUP_PAIRS = 256;
  static constexpr uint32_t HALF_LANES = 16;
  static constexpr uint32_t LOOKUP_DECODE_BYTES = 7 * RAW_ELEMENTS + 3 * DECODE_ELEMENTS +
                                                  2 * LOOKUP_PAIRS * sizeof(half) +
                                                  DECODE_COLUMNS * (HALF_LANES + 1) * sizeof(half);
  __aicore__ inline void CacheLookupWeights(int64_t tile, int64_t groups) {
    // A byte indexes a pair of exact signed FP16 nibbles. Gather copies
    // each 32-bit pair without floating-point arithmetic on its bit pattern.
    auto raw = decoder_.Get<uint8_t>();
    auto values = decoder_.Get<half>()[RAW_ELEMENTS / sizeof(half)];
    auto indices = decoder_.Get<int32_t>()[3 * RAW_ELEMENTS / sizeof(int32_t)];
    auto decoded = decoder_.Get<half>()[7 * RAW_ELEMENTS / sizeof(half)];
    auto table = decoded[DECODE_ELEMENTS];
    auto offsets = table[2 * LOOKUP_PAIRS];
    auto broadcast = offsets[DECODE_COLUMNS];
    auto expanded = broadcast[DECODE_COLUMNS * HALF_LANES].ReinterpretCast<int8_t>();
    DataCopy(table, lookup_, 2 * LOOKUP_PAIRS);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t column = 0; column < N; column += DECODE_COLUMNS) {
      for (int64_t group = 0; group < groups; ++group) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyParams codes{DECODE_COLUMNS / BLOCK, BLOCK * GROUP / 64,
                             static_cast<uint16_t>((groups - 1) * BLOCK * GROUP / 64), 0};
        DataCopy(raw.ReinterpretCast<int8_t>(), w_[((tile * N + column) / BLOCK * groups + group) * BLOCK * GROUP / 2],
                 codes);
        DataCopyParams metadata{DECODE_COLUMNS / BLOCK, 1, static_cast<uint16_t>(groups - 1), 0};
        DataCopy(offsets, zw_[((tile * N + column) / BLOCK * groups + group) * BLOCK], metadata);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(values, raw, RoundMode::CAST_NONE, RAW_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Cast(indices, values, RoundMode::CAST_RINT, RAW_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Muls(indices, indices, static_cast<int32_t>(sizeof(float)), RAW_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Gather(decoded.ReinterpretCast<float>(), table.ReinterpretCast<float>(), indices.ReinterpretCast<uint32_t>(),
               static_cast<uint32_t>(0), RAW_ELEMENTS);
        Brcb(broadcast, offsets, DECODE_COLUMNS / LANES, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t strip = 0; strip < DECODE_COLUMNS / BLOCK; ++strip) {
          for (uint32_t halfGroup = 0; halfGroup < 2; ++halfGroup) {
            const uint32_t index = (strip * 2 + halfGroup) * BLOCK * GROUP / 2;
            Sub(decoded[index], decoded[index], broadcast[strip * BLOCK * HALF_LANES], GROUP / 2, BLOCK,
                {1, 1, 0, GROUP / 2 / HALF_LANES, GROUP / 2 / HALF_LANES, 1});
          }
        }
        PipeBarrier<PIPE_V>();
        Cast(expanded, decoded, RoundMode::CAST_NONE, DECODE_ELEMENTS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams store{BLOCK, 1, 1, 0};
        for (uint32_t strip = 0; strip < DECODE_COLUMNS / BLOCK; ++strip) {
          for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
            const uint32_t source = strip * BLOCK * GROUP + kb / 2 * BLOCK * GROUP / 2 + kb % 2 * K0;
            const int64_t destination = (((column / BLOCK + strip) * groups + group) * GROUP / K0 + kb) * BLOCK * K0;
            DataCopy(b1_.Get<int8_t>()[destination], expanded[source], store);
          }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      }
    }
    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
  }
  __aicore__ inline void CachePackedWeights(int64_t tile, int64_t k, int64_t groups) {
    // INT4->half is unsupported on dav_m200. Decode bytes with exact INT16
    // masks, then interleave FP32 planes using one immutable index vector.
    auto raw = decoder_.Get<int16_t>();
    auto phase = raw[RAW_ELEMENTS];
    auto mask = phase[RAW_ELEMENTS];
    auto planes = decoder_.Get<float>()[3 * RAW_ELEMENTS / 2];
    auto decoded = planes[DECODE_ELEMENTS];
    auto indices = decoded[DECODE_ELEMENTS].ReinterpretCast<uint32_t>();
    auto expanded = decoder_.Get<int8_t>()[6 * RAW_ELEMENTS + 12 * DECODE_ELEMENTS];
    auto offsetHalf = decoder_.Get<half>()[DECODE_BASE_BYTES / 2];
    auto offsetFloat = decoder_.Get<float>()[DECODE_BASE_BYTES / 4 + DECODE_COLUMNS / 2];
    auto offsetBroadcast = offsetFloat[DECODE_COLUMNS];
    DataCopy(indices, indices_, DECODE_ELEMENTS);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t column = 0; column < N; column += DECODE_COLUMNS) {
      for (int64_t group = 0; group < groups; ++group) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyParams codes{DECODE_COLUMNS / BLOCK, BLOCK * GROUP / 64,
                             static_cast<uint16_t>((groups - 1) * BLOCK * GROUP / 64), 0};
        DataCopy(phase.ReinterpretCast<int8_t>(),
                 w_[((tile * N + column) / BLOCK * groups + group) * BLOCK * GROUP / 2], codes);
        DataCopyParams metadata{DECODE_COLUMNS / BLOCK, 1, static_cast<uint16_t>(groups - 1), 0};
        DataCopy(offsetHalf, zw_[((tile * N + column) / BLOCK * groups + group) * BLOCK], metadata);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(raw.ReinterpretCast<half>(), phase.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, RAW_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Cast(raw, raw.ReinterpretCast<half>(), RoundMode::CAST_RINT, RAW_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint32_t field = 0; field < 2; ++field) {
          Adds(phase, raw, static_cast<int16_t>(field == 0 ? 8 : 128), RAW_ELEMENTS);
          Duplicate(mask, static_cast<int16_t>(field == 0 ? 15 : 240), RAW_ELEMENTS);
          PipeBarrier<PIPE_V>();
          And(phase, phase, mask, RAW_ELEMENTS);
          PipeBarrier<PIPE_V>();
          Cast(phase.ReinterpretCast<half>(), phase, RoundMode::CAST_NONE, RAW_ELEMENTS);
          PipeBarrier<PIPE_V>();
          if (field == 1) {
            Muls(phase.ReinterpretCast<half>(), phase.ReinterpretCast<half>(), static_cast<half>(1.0f / 16),
                 RAW_ELEMENTS);
            PipeBarrier<PIPE_V>();
          }
          Adds(phase.ReinterpretCast<half>(), phase.ReinterpretCast<half>(), static_cast<half>(-8), RAW_ELEMENTS);
          PipeBarrier<PIPE_V>();
          Cast(planes[field * RAW_ELEMENTS], phase.ReinterpretCast<half>(), RoundMode::CAST_NONE, RAW_ELEMENTS);
          PipeBarrier<PIPE_V>();
        }
        Gather(decoded, planes, indices, static_cast<uint32_t>(0), DECODE_ELEMENTS);
        Cast(offsetFloat, offsetHalf, RoundMode::CAST_NONE, DECODE_COLUMNS);
        PipeBarrier<PIPE_V>();
        Brcb(offsetBroadcast, offsetFloat, DECODE_COLUMNS / LANES, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t strip = 0; strip < DECODE_COLUMNS / BLOCK; ++strip) {
          for (uint32_t halfGroup = 0; halfGroup < 2; ++halfGroup) {
            const uint32_t index = (strip * 2 + halfGroup) * BLOCK * GROUP / 2;
            Sub(decoded[index], decoded[index], offsetBroadcast[strip * BLOCK * LANES], GROUP / 2, BLOCK,
                {1, 1, 0, 8, 8, 1});
          }
        }
        PipeBarrier<PIPE_V>();
        Cast(planes.ReinterpretCast<half>(), decoded, RoundMode::CAST_NONE, DECODE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Cast(expanded, planes.ReinterpretCast<half>(), RoundMode::CAST_NONE, DECODE_ELEMENTS);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams store{BLOCK, 1, 1, 0};
        for (uint32_t strip = 0; strip < DECODE_COLUMNS / BLOCK; ++strip) {
          for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
            const uint32_t source = strip * BLOCK * GROUP + kb / 2 * BLOCK * GROUP / 2 + kb % 2 * K0;
            const int64_t destination = (((column / BLOCK + strip) * groups + group) * GROUP / K0 + kb) * BLOCK * K0;
            DataCopy(b1_.Get<int8_t>()[destination], expanded[source], store);
          }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      }
    }
    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
  }
  TPipe pipe_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> packed_, integer_, product_, metadata_, result_, decoder_;
  GlobalTensor<int8_t> x_, w_;
  GlobalTensor<half> sw_, y_, zw_, lookup_;
  GlobalTensor<float> sx_;
  GlobalTensor<uint32_t> indices_;
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_int8_projection_probe_v1(GM_ADDR x, GM_ADDR w, GM_ADDR sw, GM_ADDR sx,
                                                                    GM_ADDR output, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<> operation;
  operation.Run(x, w, sw, sx, output, descriptor);
}

extern "C" __global__ __aicore__ void qwen_w4_int8_projection_probe_v1(GM_ADDR x, GM_ADDR w, GM_ADDR sw, GM_ADDR zw,
                                                                       GM_ADDR sx, GM_ADDR output, GM_ADDR descriptor,
                                                                       GM_ADDR indices) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<true> operation;
  operation.Run(x, w, sw, sx, output, descriptor, zw, indices);
}

extern "C" __global__ __aicore__ void qwen_int8_projection_probe_m32_v1(GM_ADDR x, GM_ADDR w, GM_ADDR sw, GM_ADDR sx,
                                                                        GM_ADDR output, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<false, 32> operation;
  operation.Run(x, w, sw, sx, output, descriptor);
}

extern "C" __global__ __aicore__ void qwen_w4_int8_lookup_probe_v1(GM_ADDR x, GM_ADDR w, GM_ADDR sw, GM_ADDR zw,
                                                                   GM_ADDR sx, GM_ADDR output, GM_ADDR descriptor,
                                                                   GM_ADDR indices, GM_ADDR lookup) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<true, 16, true> operation;
  operation.Run(x, w, sw, sx, output, descriptor, zw, indices, lookup);
}
extern "C" __global__ __aicore__ void qwen_w4_int8_lookup_probe_m32_v1(GM_ADDR x, GM_ADDR w, GM_ADDR sw, GM_ADDR zw,
                                                                       GM_ADDR sx, GM_ADDR output, GM_ADDR descriptor,
                                                                       GM_ADDR indices, GM_ADDR lookup) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<true, 32, true> operation;
  operation.Run(x, w, sw, sx, output, descriptor, zw, indices, lookup);
}
