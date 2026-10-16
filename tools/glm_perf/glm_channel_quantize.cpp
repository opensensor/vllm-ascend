// SPDX-License-Identifier: Apache-2.0
// One quantization per activation group; optional SwiGLU and expert smoothing.
#include "kernel_operator.h"
#include "glm_channel_dma.h"
namespace {
using namespace AscendC;
constexpr uint32_t MAX_K = 4096, LANES = 8, FLOAT_LANES = 64;
class ChannelQuantizer {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR smoothing, GM_ADDR ends, GM_ADDR order, GM_ADDR ids, GM_ADDR low,
                             GM_ADDR high, GM_ADDR scales, GM_ADDR offsets, const __gm__ int64_t* config) {
    const int64_t rows = config[0], k = config[1], group = config[2], bits = config[3], hidden = config[4];
    const int64_t experts = config[6], expertOffset = config[7];
    if (rows < 1 || k < 256 || k > MAX_K || group < 256 || group > k || group % 256 || k % group ||
        (bits != 4 && bits != 8) || (hidden != 0 && hidden != 1) || experts < 1)
      return;
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    smoothing_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smoothing));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ends));
    order_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(order));
    ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids));
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    pipe_.InitBuffer(half_, 2 * MAX_K * sizeof(half));
    pipe_.InitBuffer(vectors_, 5 * MAX_K * sizeof(float));
    pipe_.InitBuffer(reduce_, 4 * MAX_K / FLOAT_LANES * sizeof(float));
    pipe_.InitBuffer(indices_, (MAX_K / FLOAT_LANES + LANES) * sizeof(uint32_t));
    pipe_.InitBuffer(scalar_, 4 * LANES * sizeof(float));
    pipe_.InitBuffer(mask_, 32);
    pipe_.InitBuffer(integers_, MAX_K * sizeof(int32_t));
    pipe_.InitBuffer(limb_, MAX_K * sizeof(half));
    pipe_.InitBuffer(packed_, MAX_K);
    auto indices = indices_.Get<uint32_t>();
    offsets_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    GlmChannelDma::CopyIn(indices, offsets_, MAX_K / FLOAT_LANES + LANES);
    GlmChannelDma::Fence();
    const int64_t groups = k / group;
    // Device metadata only. Peer/zero routes never read unwritten gate output.
    const int64_t validRows = hidden ? ends_.GetValue(experts - 1) : rows;
    // Prefill owns a whole row: resolve its expert once for every K group.
    // Tiny decode still distributes groups across cores instead of serializing
    // an entire token on one core. No numeric factor uses scalar GetValue.
    const int64_t groupBatch = validRows >= GetBlockNum() ? groups : 1;
    const int64_t tasksPerRow = groups / groupBatch;
    for (int64_t task = GetBlockIdx(); task < validRows * tasksPerRow; task += GetBlockNum()) {
      const int64_t row = task / tasksPerRow;
      int64_t owner = 0;
      if (hidden) {
        owner = ids_.GetValue(order_.GetValue(row)) - expertOffset;
        if (owner < 0 || owner >= experts) return;
      }
      for (int64_t groupIndex = task % tasksPerRow * groupBatch; groupIndex < (task % tasksPerRow + 1) * groupBatch;
           ++groupIndex) {
        const int64_t first = groupIndex * group;
        auto inputHalf = half_.Get<half>();
        auto upHalf = inputHalf[MAX_K];
        auto value = vectors_.Get<float>();
        auto temporary = value[MAX_K];
        auto up = value[2 * MAX_K];
        auto diagonal = value[3 * MAX_K];
        auto quant = value[4 * MAX_K];
        GlmChannelDma::CopyIn(inputHalf, input_[(hidden ? row * 2 * k : row * k) + first], group);
#ifndef GLM_CHANNEL_IDENTITY_SMOOTHING
        GlmChannelDma::CopyIn(diagonal, smoothing_[owner * k + first], group);
#endif
        if (hidden) GlmChannelDma::CopyIn(upHalf, input_[row * 2 * k + k + first], group);
        GlmChannelDma::Fence();
        Cast(value, inputHalf, RoundMode::CAST_NONE, group);
        PipeBarrier<PIPE_V>();
        if (hidden) {
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
        auto pairs = reduce_.Get<float>();
        auto maxima = pairs[2 * MAX_K / FLOAT_LANES];
        WholeReduceMax(pairs, temporary, FLOAT_LANES, repeats, 1, 1, FLOAT_LANES / LANES,
                       ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        Gather(maxima, pairs, indices, static_cast<uint32_t>(0), repeats);
        PipeBarrier<PIPE_V>();
        auto scale = scalar_.Get<float>();
        auto maximum = scale[LANES];
        auto ones = scale[2 * LANES];
        auto limit = scale[3 * LANES];
        WholeReduceMax(maximum, maxima, repeats, 1, 1, 1, FLOAT_LANES / LANES, ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        Gather(scale, maximum, indices[MAX_K / FLOAT_LANES], static_cast<uint32_t>(0), LANES);
        Duplicate(limit, bits == 4 ? 7.0f : 127.0f, LANES);
        Duplicate(ones, 1.0f, LANES);
        PipeBarrier<PIPE_V>();
        Div(scale, scale, limit, LANES);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask_.Get<uint8_t>(), scale, 0.0f, CMPMODE::EQ, LANES);
        PipeBarrier<PIPE_V>();
        Select(scale, mask_.Get<uint8_t>(), ones, scale, SELMODE::VSEL_TENSOR_TENSOR_MODE, LANES);
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
        Cast(integers_.Get<int32_t>(), quant, RoundMode::CAST_RINT, group);
        PipeBarrier<PIPE_V>();
        Cast(quant, integers_.Get<int32_t>(), RoundMode::CAST_NONE, group);
        PipeBarrier<PIPE_V>();
        Mins(quant, quant, bits == 4 ? 7.0f : 127.0f, group);
        PipeBarrier<PIPE_V>();
        Maxs(quant, quant, bits == 4 ? -7.0f : -127.0f, group);
        PipeBarrier<PIPE_V>();
        auto limb = limb_.Get<half>();
        auto packed = packed_.Get<int8_t>();
        if (bits == 8) {
          Muls(temporary, quant, 1.0f / 16.0f, group);
          PipeBarrier<PIPE_V>();
          Cast(integers_.Get<int32_t>(), temporary, RoundMode::CAST_FLOOR, group);
          PipeBarrier<PIPE_V>();
          Cast(temporary, integers_.Get<int32_t>(), RoundMode::CAST_NONE, group);
          PipeBarrier<PIPE_V>();
          Cast(limb, temporary, RoundMode::CAST_NONE, group);
          PipeBarrier<PIPE_V>();
          Cast(packed.ReinterpretCast<int4b_t>(), limb, RoundMode::CAST_NONE, group);
          GlmChannelDma::Fence();
          DataCopy(high_[row * k / 2 + first / 2], packed, group / 2);
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
        DataCopy(low_[row * k / 2 + first / 2], packed, group / 2);
        DataCopy(scales_[(row * groups + groupIndex) * LANES], scale, LANES);
        GlmChannelDma::Fence();
      }
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> half_, vectors_, reduce_, indices_, scalar_, mask_, integers_, limb_, packed_;
  GlobalTensor<half> input_;
  GlobalTensor<float> smoothing_, scales_;
  GlobalTensor<uint32_t> offsets_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<int32_t> ends_;
  GlobalTensor<int64_t> order_, ids_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_channel_quantize_v1(GM_ADDR input, GM_ADDR smoothing, GM_ADDR ends,
                                                              GM_ADDR order, GM_ADDR ids, GM_ADDR low, GM_ADDR high,
                                                              GM_ADDR scales, GM_ADDR offsets, GM_ADDR config) {
  AscendC::InitSocState();
  ChannelQuantizer operation;
  operation.Run(input, smoothing, ends, order, ids, low, high, scales, offsets,
                reinterpret_cast<__gm__ int64_t*>(config));
}
