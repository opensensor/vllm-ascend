// SPDX-License-Identifier: Apache-2.0
// Experimental disjoint expert partitions: small INT4 groups and large INT8 groups.
#include "native_int4_schedule.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"
#include "native_int8_projection_probe.cpp"
#include "native_channel_activation.cpp"

extern "C" __global__ __aicore__ void qwen_hybrid_small_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                                   GM_ADDR codes, GM_ADDR scale, GM_ADDR offset,
                                                                   GM_ADDR weightSum, GM_ADDR ends, GM_ADDR y,
                                                                   GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  native_int4::Schedule<32, 160, false, false, false, false, 128> operation;
  operation.Init(low, high, xs, sums, codes, scale, offset, weightSum, ends, nullptr, y, td);
  operation.Process();
}
extern "C" __global__ __aicore__ void qwen_hybrid_large_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR codes,
                                                                   GM_ADDR scale, GM_ADDR offset, GM_ADDR ends,
                                                                   GM_ADDR lookup, GM_ADDR y, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<true, 32, true, true> operation;
  operation.Run(low, codes, scale, xs, y, descriptor, offset, nullptr, lookup, high, ends);
}

extern "C" __global__ __aicore__ void qwen_hybrid_a8_prefill_v1(GM_ADDR x, GM_ADDR unused, GM_ADDR xs, GM_ADDR codes,
                                                                GM_ADDR scale, GM_ADDR offset, GM_ADDR ends,
                                                                GM_ADDR lookup, GM_ADDR y, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<true, 32, true, true, false> operation;
  operation.Run(x, codes, scale, xs, y, descriptor, offset, nullptr, lookup, unused, ends);
}

extern "C" __global__ __aicore__ void qwen_unpack_a8_prefill_v1(GM_ADDR low, GM_ADDR high, GM_ADDR lookup, GM_ADDR y,
                                                                GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  constexpr uint32_t BYTES = 2048, ELEMENTS = 2 * BYTES, TABLE_HALVES = 512;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, 17 * BYTES + TABLE_HALVES * sizeof(half));
  auto raw = storage.Get<uint8_t>();
  auto values = storage.Get<half>()[BYTES / sizeof(half)];
  auto indices = storage.Get<int32_t>()[3 * BYTES / sizeof(int32_t)];
  auto l = storage.Get<half>()[7 * BYTES / sizeof(half)];
  auto h = l[ELEMENTS];
  auto table = h[ELEMENTS];
  auto out = table[TABLE_HALVES].ReinterpretCast<int8_t>();
  AscendC::GlobalTensor<int8_t> lo, hi, output;
  AscendC::GlobalTensor<half> lut;
  lo.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
  hi.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
  lut.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(lookup));
  const int64_t bytes = reinterpret_cast<__gm__ int64_t*>(descriptor)[0];
  DataCopy(table, lut, TABLE_HALVES);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  for (int64_t first = GetBlockIdx() * BYTES; first < bytes; first += GetBlockNum() * BYTES) {
    const uint32_t count = bytes - first < BYTES ? bytes - first : BYTES;
    for (uint32_t limb = 0; limb < 2; ++limb) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      if (limb == 0)
        DataCopy(raw.ReinterpretCast<int8_t>(), lo[first], count);
      else
        DataCopy(raw.ReinterpretCast<int8_t>(), hi[first], count);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(values, raw, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      Cast(indices, values, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
      Muls(indices, indices, static_cast<int32_t>(sizeof(float)), count);
      PipeBarrier<PIPE_V>();
      auto decoded = limb == 0 ? l : h;
      Gather(decoded.ReinterpretCast<float>(), table.ReinterpretCast<float>(), indices.ReinterpretCast<uint32_t>(),
             static_cast<uint32_t>(0), count);
      PipeBarrier<PIPE_V>();
    }
    Muls(h, h, static_cast<half>(16), 2 * count);
    PipeBarrier<PIPE_V>();
    Add(l, l, h, 2 * count);
    PipeBarrier<PIPE_V>();
    Adds(l, l, static_cast<half>(8), 2 * count);
    PipeBarrier<PIPE_V>();
    Cast(out, l, RoundMode::CAST_NONE, 2 * count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[2 * first], out, 2 * count);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  pipe.Destroy();
}

extern "C" __global__ __aicore__ void qwen_channel_int8_projection_v1(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                                                                      GM_ADDR xs, GM_ADDR ends, GM_ADDR y,
                                                                      GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<false, 32, false, true, false, true> operation;
  operation.Run(x, weight, scale, xs, y, descriptor, nullptr, nullptr, nullptr, nullptr, ends);
}

extern "C" __global__ __aicore__ void qwen_channel_int8_projection_m64_v1(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                                                                          GM_ADDR xs, GM_ADDR ends, GM_ADDR y,
                                                                          GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<false, 64, false, true, false, true> operation;
  operation.Run(x, weight, scale, xs, y, descriptor, nullptr, nullptr, nullptr, nullptr, ends);
}

extern "C" __global__ __aicore__ void qwen_channel_int8_projection_m64_n64_v1(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                                                                              GM_ADDR xs, GM_ADDR ends, GM_ADDR y,
                                                                              GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<false, 64, false, true, false, true, 64> operation;
  operation.Run(x, weight, scale, xs, y, descriptor, nullptr, nullptr, nullptr, nullptr, ends);
}

extern "C" __global__ __aicore__ void qwen_channel_int8_integer_debug_v1(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                                                                         GM_ADDR xs, GM_ADDR ends, GM_ADDR y,
                                                                         GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<false, 32, false, true, false, true, 160, true> operation;
  operation.Run(x, weight, scale, xs, y, descriptor, nullptr, nullptr, nullptr, nullptr, ends);
}

extern "C" __global__ __aicore__ void qwen_packed_channel_int8_projection_v1(GM_ADDR x, GM_ADDR codes,
                                                                             GM_ADDR groupScales, GM_ADDR offsets,
                                                                             GM_ADDR channelScales, GM_ADDR xs,
                                                                             GM_ADDR ends, GM_ADDR lookup, GM_ADDR y,
                                                                             GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  Int8ProjectionProbe<true, 32, true, true, false, true> operation;
  operation.Run(x, codes, channelScales, xs, y, descriptor, offsets, nullptr, lookup, nullptr, ends, groupScales);
}
