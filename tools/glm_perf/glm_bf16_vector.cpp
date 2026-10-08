// SPDX-License-Identifier: Apache-2.0
// Vector BF16 storage conversion and rounding, with bounded scalar DMA tails.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t TILE = 1024;
constexpr int64_t REPEAT_ELEMENTS = 128;
class VectorBf16Convert {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, int64_t elements, int64_t mode) {
    GlobalTensor<uint32_t> input32, output32;
    GlobalTensor<uint16_t> input16, output16;
    input32.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(input));
    input16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    output32.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    output16.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output));
    pipe_.InitBuffer(source_, TILE * 4);
    pipe_.InitBuffer(words_, TILE * 4);
    pipe_.InitBuffer(scratch_, TILE * 4);
    pipe_.InitBuffer(constant_, TILE * 4);
    pipe_.InitBuffer(parity_, TILE * 4);
    pipe_.InitBuffer(result_, TILE * 4);
    pipe_.InitBuffer(output_, TILE * 2);
    pipe_.InitBuffer(canonical_, TILE * 2);
    pipe_.InitBuffer(mask_, TILE / 8);
    pipe_.InitBuffer(exponentMask_, TILE / 8);
    pipe_.InitBuffer(bitmap_, 32);
    auto source32 = source_.Get<int32_t>();
    auto source16 = source_.Get<int16_t>();
    auto words = words_.Get<int32_t>();
    auto floats = scratch_.Get<float>();
    auto constant = constant_.Get<int32_t>();
    auto parity = parity_.Get<int32_t>();
    auto result32 = result_.Get<int32_t>();
    auto result16 = output_.Get<uint16_t>();
    auto canonical = canonical_.Get<uint16_t>();
    auto nanMask = mask_.Get<uint8_t>();
    auto exponentMask = exponentMask_.Get<uint8_t>();
    auto bitmap = bitmap_.Get<uint16_t>();
    Duplicate(bitmap, static_cast<uint16_t>(0xaaaaU), 16);
    PipeBarrier<PIPE_V>();
    const bool shortInput = mode == 1;
    const bool wideOutput = mode == 1 || mode == 4;
    const int32_t inputAlignment = shortInput ? 16 : 8;
    const int32_t outputAlignment = wideOutput ? 8 : 16;
    for (int64_t offset = GetBlockIdx() * TILE; offset < elements; offset += GetBlockNum() * TILE) {
      const int32_t count = elements - offset < TILE ? elements - offset : TILE;
      const int32_t aligned = (count + REPEAT_ELEMENTS - 1) / REPEAT_ELEMENTS * REPEAT_ELEMENTS;
      const int32_t loaded = count / inputAlignment * inputAlignment;
      const int32_t stored = (count + outputAlignment - 1) / outputAlignment * outputAlignment;
      Duplicate(source32, static_cast<int32_t>(0), aligned);
      PipeBarrier<PIPE_ALL>();
      if (loaded) {
        if (shortInput)
          DataCopy(source16.ReinterpretCast<uint16_t>(), input16[offset], loaded);
        else
          DataCopy(source32.ReinterpretCast<uint32_t>(), input32[offset], loaded);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      }
      for (int32_t i = loaded; i < count; ++i) {
        if (shortInput)
          source16.ReinterpretCast<uint16_t>().SetValue(i, input16.GetValue(offset + i));
        else
          source32.ReinterpretCast<uint32_t>().SetValue(i, input32.GetValue(offset + i));
      }
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      if (shortInput) {
        // dav-2002 has no reliable direct INT16 -> FP32 vector conversion.
        // Two exact half conversions reconstruct the signed integer: the
        // upper byte is a multiple of 256 and the lower byte is at most 255.
        auto fields = words_.Get<int16_t>();
        auto halves = scratch_.Get<half>();
        Duplicate(constant.ReinterpretCast<int16_t>(), static_cast<int16_t>(0xff00U), aligned);
        PipeBarrier<PIPE_V>();
        And(fields, source16, constant.ReinterpretCast<int16_t>(), aligned);
        PipeBarrier<PIPE_V>();
        Cast(halves, fields, RoundMode::CAST_NONE, aligned);
        PipeBarrier<PIPE_V>();
        Cast(parity.ReinterpretCast<float>(), halves, RoundMode::CAST_NONE, aligned);
        Duplicate(constant.ReinterpretCast<int16_t>(), static_cast<int16_t>(0x00ff), aligned);
        PipeBarrier<PIPE_V>();
        And(fields, source16, constant.ReinterpretCast<int16_t>(), aligned);
        PipeBarrier<PIPE_V>();
        Cast(halves, fields, RoundMode::CAST_NONE, aligned);
        PipeBarrier<PIPE_V>();
        Cast(words.ReinterpretCast<float>(), halves, RoundMode::CAST_NONE, aligned);
        PipeBarrier<PIPE_V>();
        Add(floats, parity.ReinterpretCast<float>(), words.ReinterpretCast<float>(), aligned);
        PipeBarrier<PIPE_V>();
        Muls(floats, floats, 65536.0F, aligned);
        PipeBarrier<PIPE_V>();
        Cast(result32, floats, RoundMode::CAST_TRUNC, aligned);
      } else {
        Duplicate(constant, static_cast<int32_t>(0x00010000), aligned);
        PipeBarrier<PIPE_V>();
        And(parity.ReinterpretCast<uint16_t>(), source32.ReinterpretCast<uint16_t>(),
            constant.ReinterpretCast<uint16_t>(), aligned * 2);
        PipeBarrier<PIPE_V>();
        Cast(floats, parity, RoundMode::CAST_NONE, aligned);
        PipeBarrier<PIPE_V>();
        Muls(floats, floats, 1.0F / 65536.0F, aligned);
        PipeBarrier<PIPE_V>();
        Cast(parity, floats, RoundMode::CAST_TRUNC, aligned);
        Duplicate(constant, static_cast<int32_t>(0x7fff), aligned);
        PipeBarrier<PIPE_V>();
        Add(words, source32, constant, aligned);
        PipeBarrier<PIPE_V>();
        Add(words, words, parity, aligned);
        PipeBarrier<PIPE_V>();
        // Mask low words for BF16-rounded FP32 without changing storage dtype.
        Duplicate(constant, static_cast<int32_t>(0xffff0000U), aligned);
        PipeBarrier<PIPE_V>();
        And(words.ReinterpretCast<uint16_t>(), words.ReinterpretCast<uint16_t>(), constant.ReinterpretCast<uint16_t>(),
            aligned * 2);
        PipeBarrier<PIPE_V>();
        // Classify raw fields using finite numeric values, never NE(NaN,NaN).
        Duplicate(constant, static_cast<int32_t>(0x7f800000), aligned);
        PipeBarrier<PIPE_V>();
        And(parity.ReinterpretCast<uint16_t>(), source32.ReinterpretCast<uint16_t>(),
            constant.ReinterpretCast<uint16_t>(), aligned * 2);
        PipeBarrier<PIPE_V>();
        Cast(floats, parity, RoundMode::CAST_NONE, aligned);
        PipeBarrier<PIPE_V>();
        CompareScalar(exponentMask, floats, static_cast<float>(0x7f800000), CMPMODE::EQ, aligned);
        Duplicate(constant, static_cast<int32_t>(0x007fffff), aligned);
        PipeBarrier<PIPE_V>();
        And(parity.ReinterpretCast<uint16_t>(), source32.ReinterpretCast<uint16_t>(),
            constant.ReinterpretCast<uint16_t>(), aligned * 2);
        PipeBarrier<PIPE_V>();
        Cast(floats, parity, RoundMode::CAST_NONE, aligned);
        PipeBarrier<PIPE_V>();
        CompareScalar(nanMask, floats, 0.0F, CMPMODE::GT, aligned);
        PipeBarrier<PIPE_V>();
        And(nanMask.ReinterpretCast<uint16_t>(), nanMask.ReinterpretCast<uint16_t>(),
            exponentMask.ReinterpretCast<uint16_t>(), aligned / 16);
        Duplicate(constant, static_cast<int32_t>(0x80000000U), aligned);
        PipeBarrier<PIPE_V>();
        And(parity.ReinterpretCast<uint16_t>(), source32.ReinterpretCast<uint16_t>(),
            constant.ReinterpretCast<uint16_t>(), aligned * 2);
        Duplicate(constant, static_cast<int32_t>(0x7fc00000), aligned);
        PipeBarrier<PIPE_V>();
        Or(parity.ReinterpretCast<uint16_t>(), parity.ReinterpretCast<uint16_t>(), constant.ReinterpretCast<uint16_t>(),
           aligned * 2);
        PipeBarrier<PIPE_V>();
        Select(result32.ReinterpretCast<float>(), nanMask, parity.ReinterpretCast<float>(),
               words.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
        PipeBarrier<PIPE_V>();
        if (mode == 0) {
          uint64_t reserved = 0;
          GatherMask(result16, result32.ReinterpretCast<uint16_t>(), bitmap, true, aligned * 2, {1, 1, 8, 0}, reserved);
        } else if (mode == 5) {
          // The hardware rounding/subnormal contract is gated exhaustively.
          Cast(result16.ReinterpretCast<half>(), result32.ReinterpretCast<float>(), RoundMode::CAST_NONE, aligned);
          PipeBarrier<PIPE_V>();
          uint64_t reserved = 0;
          GatherMask(canonical, source32.ReinterpretCast<uint16_t>(), bitmap, true, aligned * 2, {1, 1, 8, 0},
                     reserved);
          PipeBarrier<PIPE_V>();
          Duplicate(constant.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x8000U), aligned);
          PipeBarrier<PIPE_V>();
          And(canonical, canonical, constant.ReinterpretCast<uint16_t>(), aligned);
          Duplicate(constant.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x7e00U), aligned);
          PipeBarrier<PIPE_V>();
          Or(canonical, canonical, constant.ReinterpretCast<uint16_t>(), aligned);
          PipeBarrier<PIPE_V>();
          Select(result16.ReinterpretCast<half>(), nanMask, canonical.ReinterpretCast<half>(),
                 result16.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
        }
      }
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      if (wideOutput)
        DataCopy(output32[offset], result32.ReinterpretCast<uint32_t>(), stored);
      else
        DataCopy(output16[offset], result16, stored);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> source_, words_, scratch_, constant_, parity_, result_, output_, canonical_, mask_,
      exponentMask_, bitmap_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_bf16_vector_v1(GM_ADDR input, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  auto values = reinterpret_cast<__gm__ int64_t*>(config);
  if (values[0] <= 0 || (values[1] != 0 && values[1] != 1 && values[1] != 4 && values[1] != 5)) return;
  VectorBf16Convert converter;
  converter.Run(input, output, values[0], values[1]);
}
