// SPDX-License-Identifier: Apache-2.0
// True UB gate/up -> SwiGLU -> hidden quant; no full gate/up route tensor.
#define GLM_CHANNEL_FUSED_GATE_UP
#include "glm_channel_projection.cpp"
#include "glm_channel_dma.h"
namespace {
constexpr uint32_t HIDDEN_GROUP = 256, FLOAT_LANES = 64;
class FusedGateUp : public Projection {
 public:
  __aicore__ inline void RunFused(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes, GM_ADDR scales, GM_ADDR sums,
                                  GM_ADDR ends, GM_ADDR order, GM_ADDR sortedKeys, GM_ADDR productOffsets,
                                  GM_ADDR smoothing, GM_ADDR hiddenLow, GM_ADDR hiddenHigh, GM_ADDR hiddenScales,
                                  GM_ADDR quantOffsets, const __gm__ int64_t* config) {
    if (!Init(low, high, xs, codes, scales, sums, ends, order, sortedKeys, productOffsets, hiddenLow, config)) return;
    intermediate_ = n_ / 2;
    if (intermediate_ % HIDDEN_GROUP || !originalTokens_) return;
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing));
    hiddenLow_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(hiddenLow));
    hiddenHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(hiddenHigh));
    hiddenScales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hiddenScales));
    quantOffsets_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(quantOffsets));
    const uint32_t maxRows = bits_ == 8 ? ROWS / 2 : ROWS;
    pipe_.InitBuffer(hidden_, maxRows * 2 * HIDDEN_GROUP * sizeof(half));
    pipe_.InitBuffer(selectMask_, 32);
    for (int64_t hiddenGroup = GetBlockIdx(); hiddenGroup < intermediate_ / HIDDEN_GROUP;
         hiddenGroup += GetBlockNum()) {
      int64_t first = 0;
      const int64_t valid = ends_.GetValue(experts_ - 1);
      for (int64_t expert = 0; expert < experts_ && first < valid; ++expert) {
        if (tokens_ <= 16) expert = static_cast<int64_t>(sortedKeys_.GetValue(first));
        if (expert < 0 || expert >= experts_) return;
        const int64_t end = ends_.GetValue(expert);
        if (end < first || end > tokens_ * topK_) return;
        for (int64_t row = first; row < end; row += maxRows) {
          const uint32_t count = end - row < maxRows ? end - row : maxRows;
          for (uint32_t projection = 0; projection < 2; ++projection) {
            for (uint32_t part = 0; part < HIDDEN_GROUP / N; ++part) {
              const int64_t tile = projection * (intermediate_ / N) + hiddenGroup * (HIDDEN_GROUP / N) + part;
              ProjectBlock(row, count, expert, tile);
              for (uint32_t m = 0; m < count; ++m) {
                Adds(hidden_.Get<half>()[m * 2 * HIDDEN_GROUP + projection * HIDDEN_GROUP + part * N],
                     outputUb_.Get<half>()[m * N], static_cast<half>(0), N);
              }
              PipeBarrier<PIPE_V>();
            }
          }
          QuantizeHidden(row, count, expert, hiddenGroup);
        }
        first = end;
      }
    }
  }

 private:
  __aicore__ inline void QuantizeHidden(int64_t row, uint32_t count, int64_t expert, int64_t groupIndex) {
    const uint32_t group = HIDDEN_GROUP, bits = bits_;
    auto indices = indices_.Get<uint32_t>();
    GlmChannelDma::CopyIn(indices, quantOffsets_, 72);
#ifndef GLM_CHANNEL_IDENTITY_SMOOTHING
    GlmChannelDma::CopyIn(products_.Get<float>()[3 * group], smoothing_[expert * intermediate_ + groupIndex * group],
                          group);
#endif
    GlmChannelDma::Fence();
    for (uint32_t m = 0; m < count; ++m) {
      auto inputHalf = hidden_.Get<half>()[m * 2 * group];
      auto upHalf = inputHalf[group];
      auto value = products_.Get<float>();
      auto temporary = value[group];
      auto up = value[2 * group];
      auto diagonal = value[3 * group];
      auto quant = value[4 * group];
      Cast(value, inputHalf, RoundMode::CAST_NONE, group);
      PipeBarrier<PIPE_V>();
      {
        Cast(up, upHalf, RoundMode::CAST_NONE, group);
        Muls(temporary, value, -1.0f, group);
        PipeBarrier<PIPE_V>();
        Exp(temporary, temporary, group);
        PipeBarrier<PIPE_V>();
        Adds(temporary, temporary, 1.0f, group);
        PipeBarrier<PIPE_V>();
        Div(value, value, temporary, group);
        PipeBarrier<PIPE_V>();
        Mul(value, value, up, group);
        PipeBarrier<PIPE_V>();
        Mins(value, value, 65504.0f, group);
        PipeBarrier<PIPE_V>();
        Maxs(value, value, -65504.0f, group);
        PipeBarrier<PIPE_V>();
        Cast(inputHalf, value, RoundMode::CAST_NONE, group);
        PipeBarrier<PIPE_V>();
        Cast(value, inputHalf, RoundMode::CAST_NONE, group);
        PipeBarrier<PIPE_V>();
      }
#ifndef GLM_CHANNEL_IDENTITY_SMOOTHING
      Div(value, value, diagonal, group);
      PipeBarrier<PIPE_V>();
#endif
      Abs(temporary, value, group);
      PipeBarrier<PIPE_V>();
      const uint32_t repeats = group / FLOAT_LANES;
      auto pairs = weights_.Get<float>();
      auto maxima = pairs[N];
      WholeReduceMax(pairs, temporary, FLOAT_LANES, repeats, 1, 1, FLOAT_LANES / LANES, ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(maxima, pairs, indices, static_cast<uint32_t>(0), repeats);
      PipeBarrier<PIPE_V>();
      auto scale = sx_.Get<float>();
      auto maximum = scale[LANES];
      auto ones = scale[2 * LANES];
      auto limit = scale[3 * LANES];
      WholeReduceMax(maximum, maxima, repeats, 1, 1, 1, FLOAT_LANES / LANES, ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(scale, maximum, indices[64], static_cast<uint32_t>(0), LANES);
      Duplicate(limit, bits == 4 ? 7.0f : 127.0f, LANES);
      Duplicate(ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Div(scale, scale, limit, LANES);
      PipeBarrier<PIPE_V>();
      CompareScalar(selectMask_.Get<uint8_t>(), scale, 0.0f, CMPMODE::EQ, LANES);
      PipeBarrier<PIPE_V>();
      Select(scale, selectMask_.Get<uint8_t>(), ones, scale, SELMODE::VSEL_TENSOR_TENSOR_MODE, LANES);
      PipeBarrier<PIPE_V>();
      const BinaryRepeatParams broadcast{1, 1, 0, 8, 8, 0};
#ifdef GLM_CHANNEL_RECIPROCAL_QUANT
      Div(limit, ones, scale, LANES);
      PipeBarrier<PIPE_V>();
      Mul(quant, value, limit, static_cast<uint64_t>(FLOAT_LANES), repeats, broadcast);
#else
      Div(quant, value, scale, static_cast<uint64_t>(FLOAT_LANES), repeats, broadcast);
#endif
      PipeBarrier<PIPE_V>();
      Cast(raw_.Get<int32_t>(), quant, RoundMode::CAST_RINT, group);
      PipeBarrier<PIPE_V>();
      Cast(quant, raw_.Get<int32_t>(), RoundMode::CAST_NONE, group);
      PipeBarrier<PIPE_V>();
      Mins(quant, quant, bits == 4 ? 7.0f : 127.0f, group);
      PipeBarrier<PIPE_V>();
      Maxs(quant, quant, bits == 4 ? -7.0f : -127.0f, group);
      PipeBarrier<PIPE_V>();
      auto limb = outputUb_.Get<half>();
      auto packed = aUb_.Get<int8_t>();
      if (bits == 8) {
        Muls(temporary, quant, 1.0f / 16.0f, group);
        PipeBarrier<PIPE_V>();
        Cast(raw_.Get<int32_t>(), temporary, RoundMode::CAST_FLOOR, group);
        PipeBarrier<PIPE_V>();
        Cast(temporary, raw_.Get<int32_t>(), RoundMode::CAST_NONE, group);
        PipeBarrier<PIPE_V>();
        Cast(limb, temporary, RoundMode::CAST_NONE, group);
        PipeBarrier<PIPE_V>();
        Cast(packed.ReinterpretCast<int4b_t>(), limb, RoundMode::CAST_NONE, group);
        GlmChannelDma::Fence();
        DataCopy(hiddenHigh_[(row + m) * intermediate_ / 2 + groupIndex * group / 2], packed, group / 2);
        GlmChannelDma::Fence();
        Muls(temporary, temporary, -16.0f, group);
        PipeBarrier<PIPE_V>();
        Add(quant, quant, temporary, group);
        PipeBarrier<PIPE_V>();
        Adds(quant, quant, -8.0f, group);
        PipeBarrier<PIPE_V>();
      }
      Cast(limb, quant, RoundMode::CAST_NONE, group);
      PipeBarrier<PIPE_V>();
      Cast(packed.ReinterpretCast<int4b_t>(), limb, RoundMode::CAST_NONE, group);
      GlmChannelDma::Fence();
      DataCopy(hiddenLow_[(row + m) * intermediate_ / 2 + groupIndex * group / 2], packed, group / 2);
      DataCopy(hiddenScales_[((row + m) * (intermediate_ / group) + groupIndex) * LANES], scale, LANES);
      GlmChannelDma::Fence();
    }
  }
  TBuf<TPosition::VECCALC> hidden_, selectMask_;
  GlobalTensor<int8_t> hiddenLow_, hiddenHigh_;
  GlobalTensor<float> hiddenScales_, smoothing_;
  GlobalTensor<uint32_t> quantOffsets_;
  int64_t intermediate_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_channel_gate_up_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                             GM_ADDR scales, GM_ADDR sums, GM_ADDR ends, GM_ADDR order,
                                                             GM_ADDR sortedKeys, GM_ADDR productOffsets,
                                                             GM_ADDR smoothing, GM_ADDR hiddenLow, GM_ADDR hiddenHigh,
                                                             GM_ADDR hiddenScales, GM_ADDR quantOffsets,
                                                             GM_ADDR config) {
  AscendC::InitSocState();
  FusedGateUp operation;
  operation.RunFused(low, high, xs, codes, scales, sums, ends, order, sortedKeys, productOffsets, smoothing, hiddenLow,
                     hiddenHigh, hiddenScales, quantOffsets, reinterpret_cast<__gm__ int64_t*>(config));
}
