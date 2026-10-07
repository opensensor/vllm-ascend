// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t GROUP = 32, BATCH = 8, LANES = 8, ELEMENTS = GROUP * BATCH;
class Pack {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR low, GM_ADDR high, GM_ADDR xs, int64_t groups) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    pipe_.InitBuffer(storage_, 16384);
    pipe_.InitBuffer(padded_, 2 * ELEMENTS * sizeof(half));
    pipe_.InitBuffer(offsets_, 2 * ELEMENTS * sizeof(uint32_t));
    auto input = storage_.Get<half>();
    auto values = storage_.Get<float>()[ELEMENTS];
    auto temporary = storage_.Get<float>()[2 * ELEMENTS];
    auto quant = storage_.Get<float>()[3 * ELEMENTS];
    auto integers = storage_.Get<int32_t>()[4 * ELEMENTS];
    auto limb = storage_.Get<half>()[10 * ELEMENTS];
    auto packed = storage_.Get<int8_t>()[24 * ELEMENTS];
    auto pairs = storage_.Get<float>()[7 * ELEMENTS];
    auto maxima = pairs[2 * BATCH];
    auto scales = maxima[LANES];
    auto ones = scales[LANES];
    auto broadcast = ones[LANES];
    auto indices = broadcast[LANES * BATCH].ReinterpretCast<uint32_t>();
    auto offsets = offsets_.Get<uint32_t>();
    for (uint32_t group = 0; group < BATCH; ++group)
      for (uint32_t k = 0; k < 2 * GROUP; ++k)
        offsets.SetValue(group * 2 * GROUP + k, (k < GROUP ? group * GROUP + k : ELEMENTS) * sizeof(half));
    Duplicate(limb[ELEMENTS], static_cast<half>(0), GROUP);
    for (uint32_t i = 0; i < BATCH; ++i) indices.SetValue(i, i * 2 * sizeof(float));
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (int64_t first = GetBlockIdx() * BATCH; first < groups; first += GetBlockNum() * BATCH) {
      // K is divisible by 256, so every launch has whole eight-group batches.
      DataCopy(input, input_[first * GROUP], ELEMENTS);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(values, input, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Abs(temporary, values, ELEMENTS);
      PipeBarrier<PIPE_V>();
      WholeReduceMax(pairs, temporary, GROUP, BATCH, 1, 1, GROUP / LANES, ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(maxima, pairs, indices, static_cast<uint32_t>(0), LANES);
      Duplicate(ones, 127.0f, LANES);
      PipeBarrier<PIPE_V>();
      Div(scales, maxima, ones, LANES);
      PipeBarrier<PIPE_V>();
      Muls(ones, maxima, 16777216.0f, LANES);
      PipeBarrier<PIPE_V>();
      Mins(ones, ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Muls(ones, ones, -1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Adds(ones, ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Add(scales, scales, ones, LANES);
      PipeBarrier<PIPE_V>();
      Brcb(broadcast, scales, 1, {1, 8});
      PipeBarrier<PIPE_V>();
      Div(quant, values, broadcast, GROUP, BATCH, {1, 1, 0, GROUP / LANES, GROUP / LANES, 1});
      PipeBarrier<PIPE_V>();
      Cast(integers, quant, RoundMode::CAST_RINT, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(quant, integers, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Mins(quant, quant, 127.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Maxs(quant, quant, -127.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Muls(temporary, quant, 1.0f / 16.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(integers, temporary, RoundMode::CAST_FLOOR, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(temporary, integers, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(limb, temporary, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Gather(padded_.Get<half>(), limb, offsets, static_cast<uint32_t>(0), 2 * ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(packed.ReinterpretCast<int4b_t>(), padded_.Get<half>(), RoundMode::CAST_NONE, 2 * ELEMENTS);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      for (uint32_t group = 0; group < BATCH; ++group) DataCopy(high_[(first + group) * 32], packed[group * GROUP], 32);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      Muls(temporary, temporary, -16.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Add(temporary, quant, temporary, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Adds(temporary, temporary, -8.0f, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(limb, temporary, RoundMode::CAST_NONE, ELEMENTS);
      PipeBarrier<PIPE_V>();
      Gather(padded_.Get<half>(), limb, offsets, static_cast<uint32_t>(0), 2 * ELEMENTS);
      PipeBarrier<PIPE_V>();
      Cast(packed.ReinterpretCast<int4b_t>(), padded_.Get<half>(), RoundMode::CAST_NONE, 2 * ELEMENTS);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      for (uint32_t group = 0; group < BATCH; ++group) DataCopy(low_[(first + group) * 32], packed[group * GROUP], 32);
      DataCopy(xs_[first * LANES], broadcast, BATCH * LANES);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  TBuf<TPosition::VECCALC> padded_, offsets_;
  GlobalTensor<half> input_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<float> xs_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_w4a8_pack_v1(GM_ADDR x, GM_ADDR low, GM_ADDR high, GM_ADDR xs,
                                                       GM_ADDR tiling) {
  AscendC::InitSocState();
  Pack operation;
  operation.Run(x, low, high, xs, reinterpret_cast<__gm__ int64_t*>(tiling)[0]);
}
