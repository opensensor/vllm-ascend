// SPDX-License-Identifier: Apache-2.0
// Experimental G128-packed activation -> per-row INT8 without a full FP32 GM tensor.
#include "kernel_operator.h"
namespace channel_activation {
using namespace AscendC;
constexpr uint32_t MAX_K = 2560, MAX_ROWS = 25600, GROUP = 128, LANES = 8, LOOKUP_HALVES = 512;
constexpr uint32_t UB_BYTES = 64 * 1024;
// Reciprocal of the smallest nonzero FP16 value: distinguish zero maxima
// without device-to-host inspection or division by zero.
constexpr float FP16_MIN_SUBNORMAL_RECIPROCAL = 16777216.0f;
template <bool PACKED_INPUT = true, bool LOCAL_ROWS = false>
class ChannelActivation {
 public:
  __aicore__ inline void Run(GM_ADDR lo, GM_ADDR hi, GM_ADDR sx, GM_ADDR lut, GM_ADDR y, GM_ADDR sy, GM_ADDR config,
                             GM_ADDR ends = nullptr) {
    auto td = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t rows = td[0], k = td[1], groups = k / GROUP;
    ASCENDC_ASSERT(rows > 0 && rows <= MAX_ROWS && k > 0 && k <= MAX_K && k % GROUP == 0,
                   { KERNEL_LOG(KERNEL_ERROR, "invalid channel activation geometry"); });
    int64_t activeRows = rows;
    if constexpr (LOCAL_ROWS) {
      const int64_t experts = td[2];
      ASCENDC_ASSERT(experts > 0 && experts <= 128, { KERNEL_LOG(KERNEL_ERROR, "invalid local expert count"); });
      GlobalTensor<int64_t> groupEnds;
      groupEnds.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
      activeRows = groupEnds.GetValue(experts - 1);
      ASCENDC_ASSERT(activeRows >= 0 && activeRows <= rows, { KERNEL_LOG(KERNEL_ERROR, "invalid local row count"); });
    }
    GlobalTensor<int8_t> low, high, output;
    GlobalTensor<float> inputScale, outputScale;
    GlobalTensor<half> lookup;
    low.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(lo));
    high.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(hi));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
    inputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sx));
    outputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sy));
    lookup.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(lut));
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    // Fixed regions never alias live vector operands. Output packing reuses
    // decoded halves only after the FP32 values and maxima have been formed.
    pipe.InitBuffer(storage, UB_BYTES);
    auto raw = storage.Get<uint8_t>();
    auto values = storage.Get<half>()[MAX_K / 4];
    auto offsets = storage.Get<int32_t>()[3 * MAX_K / 8];
    auto l = storage.Get<half>()[7 * MAX_K / 4];
    auto h = l[MAX_K];
    auto table = h[MAX_K];
    auto f = storage.Get<float>()[3 * MAX_K + LOOKUP_HALVES / 2];
    auto scratch = f[MAX_K];
    auto scales = scratch[MAX_K];
    auto pairs = scales[MAX_K / GROUP * LANES];
    auto maxima = pairs[MAX_K / 32];
    auto reduceIndices = maxima[64].ReinterpretCast<uint32_t>();
    auto finalMax = reduceIndices[64].ReinterpretCast<float>();
    auto rowScale = finalMax[LANES];
    auto ones = rowScale[LANES];
    auto zeroIndices = ones[LANES].ReinterpretCast<uint32_t>();
    for (uint32_t i = 0; i < MAX_K / 64; ++i) reduceIndices.SetValue(i, 2 * i * sizeof(float));
    for (uint32_t i = 0; i < LANES; ++i) zeroIndices.SetValue(i, 0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    if constexpr (PACKED_INPUT) {
      DataCopy(table, lookup, LOOKUP_HALVES);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    for (int64_t row = GetBlockIdx(); row < activeRows; row += GetBlockNum()) {
      if constexpr (PACKED_INPUT) {
        for (uint32_t limb = 0; limb < 2; ++limb) {
          SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
          if (limb == 0)
            DataCopy(raw.ReinterpretCast<int8_t>(), low[row * k / 2], k / 2);
          else
            DataCopy(raw.ReinterpretCast<int8_t>(), high[row * k / 2], k / 2);
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
          Cast(values, raw, RoundMode::CAST_NONE, k / 2);
          PipeBarrier<PIPE_V>();
          Cast(offsets, values, RoundMode::CAST_RINT, k / 2);
          PipeBarrier<PIPE_V>();
          Muls(offsets, offsets, static_cast<int32_t>(sizeof(float)), k / 2);
          PipeBarrier<PIPE_V>();
          auto decoded = limb == 0 ? l : h;
          Gather(decoded.ReinterpretCast<float>(), table.ReinterpretCast<float>(), offsets.ReinterpretCast<uint32_t>(),
                 static_cast<uint32_t>(0), k / 2);
          PipeBarrier<PIPE_V>();
        }
        Muls(h, h, static_cast<half>(16), k);
        PipeBarrier<PIPE_V>();
        Add(l, l, h, k);
        PipeBarrier<PIPE_V>();
        Adds(l, l, static_cast<half>(8), k);
        PipeBarrier<PIPE_V>();
        Cast(f, l, RoundMode::CAST_NONE, k);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(scales, inputScale[row * groups * LANES], groups * LANES);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (int64_t group = 0; group < groups; ++group)
          Mul(f[group * GROUP], f[group * GROUP], scales[group * LANES], 64, 2, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Cast(l, f, RoundMode::CAST_NONE, k);
        PipeBarrier<PIPE_V>();
        Cast(f, l, RoundMode::CAST_NONE, k);
        PipeBarrier<PIPE_V>();
      } else {
        GlobalTensor<half> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(lo));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(l, input[row * k], k);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(f, l, RoundMode::CAST_NONE, k);
        PipeBarrier<PIPE_V>();
      }
      Abs(scratch, f, k);
      PipeBarrier<PIPE_V>();
      WholeReduceMax(pairs, scratch, 64, k / 64, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(maxima, pairs, reduceIndices, static_cast<uint32_t>(0), k / 64);
      PipeBarrier<PIPE_V>();
      WholeReduceMax(finalMax, maxima, k / 64, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(maxima, finalMax, zeroIndices, static_cast<uint32_t>(0), LANES);
      Duplicate(ones, 127.0f, LANES);
      PipeBarrier<PIPE_V>();
      Div(rowScale, maxima, ones, LANES);
      PipeBarrier<PIPE_V>();
      Muls(ones, maxima, FP16_MIN_SUBNORMAL_RECIPROCAL, LANES);
      PipeBarrier<PIPE_V>();
      Mins(ones, ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Muls(ones, ones, -1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Adds(ones, ones, 1.0f, LANES);
      PipeBarrier<PIPE_V>();
      Add(rowScale, rowScale, ones, LANES);
      PipeBarrier<PIPE_V>();
      Div(scratch, f, rowScale, 64, k / 64, {1, 1, 0, 8, 8, 0});
      PipeBarrier<PIPE_V>();
      Cast(f.ReinterpretCast<int32_t>(), scratch, RoundMode::CAST_RINT, k);
      PipeBarrier<PIPE_V>();
      Cast(scratch, f.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, k);
      PipeBarrier<PIPE_V>();
      Mins(scratch, scratch, 127.0f, k);
      PipeBarrier<PIPE_V>();
      Maxs(scratch, scratch, -127.0f, k);
      PipeBarrier<PIPE_V>();
      Cast(l, scratch, RoundMode::CAST_NONE, k);
      PipeBarrier<PIPE_V>();
      Cast(raw.ReinterpretCast<int8_t>(), l, RoundMode::CAST_NONE, k);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output[row * k], raw.ReinterpretCast<int8_t>(), k);
      DataCopy(outputScale[row * LANES], rowScale, LANES);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    pipe.Destroy();
  }
};
}  // namespace channel_activation
extern "C" __global__ __aicore__ void qwen_channel_activation_v1(GM_ADDR low, GM_ADDR high, GM_ADDR scales,
                                                                 GM_ADDR lookup, GM_ADDR output, GM_ADDR outputScale,
                                                                 GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  channel_activation::ChannelActivation<> operation;
  operation.Run(low, high, scales, lookup, output, outputScale, descriptor);
}

extern "C" __global__ __aicore__ void qwen_direct_channel_activation_v1(GM_ADDR input, GM_ADDR output,
                                                                        GM_ADDR outputScale, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  channel_activation::ChannelActivation<false> operation;
  operation.Run(input, nullptr, nullptr, nullptr, output, outputScale, descriptor);
}

extern "C" __global__ __aicore__ void qwen_direct_local_channel_activation_v1(GM_ADDR input, GM_ADDR ends,
                                                                              GM_ADDR output, GM_ADDR outputScale,
                                                                              GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  channel_activation::ChannelActivation<false, true> operation;
  operation.Run(input, nullptr, nullptr, nullptr, output, outputScale, descriptor, ends);
}
