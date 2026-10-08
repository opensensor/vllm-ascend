// SPDX-License-Identifier: Apache-2.0
// Experimental FP16 -> BF16 query conversion for dav-2002. Compile-only until
// exhaustive parity, changed replay, padding and real-model gates pass.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t TILE = 1024;
constexpr int64_t DMA_HALF_ELEMENTS = 16;
constexpr int64_t HALF_COMPARE_ELEMENTS = 128;
class QueryConvert {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, int64_t elements) {
    GlobalTensor<half> source;
    GlobalTensor<uint16_t> sourceBits;
    GlobalTensor<uint16_t> destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    sourceBits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output));
    pipe_.InitBuffer(input_, TILE * sizeof(half));
    pipe_.InitBuffer(words_, TILE * sizeof(float));
    pipe_.InitBuffer(parity_, TILE * sizeof(int32_t));
    pipe_.InitBuffer(floats_, TILE * sizeof(float));
    pipe_.InitBuffer(constant_, TILE * sizeof(int32_t));
    pipe_.InitBuffer(rounded_, TILE * sizeof(int32_t));
    pipe_.InitBuffer(output_, TILE * sizeof(uint16_t));
    pipe_.InitBuffer(canonical_, TILE * sizeof(uint16_t));
    pipe_.InitBuffer(nanMask_, TILE / 8);
    pipe_.InitBuffer(exponentMask_, TILE / 8);
    pipe_.InitBuffer(gatherMask_, 32);
    auto inputLocal = input_.Get<half>();
    auto wordsFloat = words_.Get<float>();
    auto wordsInt = words_.Get<int32_t>();
    auto parityInt = parity_.Get<int32_t>();
    auto parityFloat = floats_.Get<float>();
    auto constantInt = constant_.Get<int32_t>();
    auto roundedInt = rounded_.Get<int32_t>();
    auto result = output_.Get<uint16_t>();
    auto canonical = canonical_.Get<uint16_t>();
    auto nanMask = nanMask_.Get<uint8_t>();
    auto exponentMask = exponentMask_.Get<uint8_t>();
    auto gatherMask = gatherMask_.Get<uint16_t>();
    // Fixed odd-word bitmap extracts the high 16 bits of each FP32 word.
    // It repeats from the same 32-byte bank, rather than relying on a magic
    // gather pattern number or an unsupported dav-2002 vector shift.
    Duplicate(gatherMask, static_cast<uint16_t>(0xaaaaU), 16);
    PipeBarrier<PIPE_V>();
    for (int64_t offset = GetBlockIdx() * TILE; offset < elements; offset += GetBlockNum() * TILE) {
      const int32_t count = elements - offset < TILE ? elements - offset : TILE;
      const int32_t loaded = count / DMA_HALF_ELEMENTS * DMA_HALF_ELEMENTS;
      const int32_t dmaAligned = (count + DMA_HALF_ELEMENTS - 1) / DMA_HALF_ELEMENTS * DMA_HALF_ELEMENTS;
      // dav-m200 Compare counts truncate to whole 256-byte repeats. Fully
      // initialize a whole half repeat even when the owned GM tail is shorter.
      const int32_t aligned = (count + HALF_COMPARE_ELEMENTS - 1) / HALF_COMPARE_ELEMENTS * HALF_COMPARE_ELEMENTS;
      Duplicate(inputLocal, static_cast<half>(0), aligned);
      PipeBarrier<PIPE_ALL>();
      if (loaded) {
        DataCopy(inputLocal, source[offset], loaded);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      }
      // At most 15 source elements use scalar tail reads. The input need not
      // own DMA padding; the output explicitly does. No whole-tile scalar loop.
      for (int32_t i = loaded; i < dmaAligned; ++i)
        inputLocal.ReinterpretCast<uint16_t>().SetValue(i, i < count ? sourceBits.GetValue(offset + i) : 0);
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      Cast(wordsFloat, inputLocal, RoundMode::CAST_NONE, aligned);
      PipeBarrier<PIPE_V>();
      Duplicate(constantInt, static_cast<int32_t>(0x00010000), aligned);
      PipeBarrier<PIPE_V>();
      And(parityInt.ReinterpretCast<uint16_t>(), wordsInt.ReinterpretCast<uint16_t>(),
          constantInt.ReinterpretCast<uint16_t>(), aligned * 2);
      PipeBarrier<PIPE_V>();
      // Extract bit 16 using exact numeric conversion of {0,65536}. Scalar
      // shifts are unavailable as vector APIs on this target; float scaling
      // by 2^-16 is exact for these two values.
      Cast(parityFloat, parityInt, RoundMode::CAST_NONE, aligned);
      PipeBarrier<PIPE_V>();
      Muls(parityFloat, parityFloat, 1.0F / 65536.0F, aligned);
      PipeBarrier<PIPE_V>();
      Cast(parityInt, parityFloat, RoundMode::CAST_TRUNC, aligned);
      Duplicate(constantInt, static_cast<int32_t>(0x7fff), aligned);
      PipeBarrier<PIPE_V>();
      Add(roundedInt, wordsInt, constantInt, aligned);
      PipeBarrier<PIPE_V>();
      Add(roundedInt, roundedInt, parityInt, aligned);
      PipeBarrier<PIPE_V>();
      uint64_t reserved = 0;
      GatherMask(result, roundedInt.ReinterpretCast<uint16_t>(), gatherMask, true, aligned * 2, {1, 1, 8, 0}, reserved);
      PipeBarrier<PIPE_V>();
      // NE(x,x) does not classify NaNs reliably on this target. Classify raw
      // exponent and mantissa using finite, exactly representable half integers.
      auto fields = parity_.Get<int16_t>();
      auto finiteFields = floats_.Get<half>();
      Duplicate(constantInt.ReinterpretCast<int16_t>(), static_cast<int16_t>(0x7c00), aligned);
      PipeBarrier<PIPE_V>();
      And(fields, inputLocal.ReinterpretCast<int16_t>(), constantInt.ReinterpretCast<int16_t>(), aligned);
      PipeBarrier<PIPE_V>();
      Cast(finiteFields, fields, RoundMode::CAST_NONE, aligned);
      PipeBarrier<PIPE_V>();
      CompareScalar(exponentMask, finiteFields, static_cast<half>(31744), CMPMODE::EQ, aligned);
      Duplicate(constantInt.ReinterpretCast<int16_t>(), static_cast<int16_t>(0x03ff), aligned);
      PipeBarrier<PIPE_V>();
      And(fields, inputLocal.ReinterpretCast<int16_t>(), constantInt.ReinterpretCast<int16_t>(), aligned);
      PipeBarrier<PIPE_V>();
      Cast(finiteFields, fields, RoundMode::CAST_NONE, aligned);
      PipeBarrier<PIPE_V>();
      CompareScalar(nanMask, finiteFields, static_cast<half>(0), CMPMODE::GT, aligned);
      PipeBarrier<PIPE_V>();
      And(nanMask.ReinterpretCast<uint16_t>(), nanMask.ReinterpretCast<uint16_t>(),
          exponentMask.ReinterpretCast<uint16_t>(), aligned / 16);
      PipeBarrier<PIPE_V>();
      // Preserve the existing signed canonical BF16 NaN bits.
      Duplicate(canonical, static_cast<uint16_t>(0x8000U), aligned);
      PipeBarrier<PIPE_V>();
      And(canonical, inputLocal.ReinterpretCast<uint16_t>(), canonical, aligned);
      Duplicate(constantInt.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x7fc0U), aligned);
      PipeBarrier<PIPE_V>();
      Or(canonical, canonical, constantInt.ReinterpretCast<uint16_t>(), aligned);
      PipeBarrier<PIPE_V>();
      Select(result.ReinterpretCast<half>(), nanMask, canonical.ReinterpretCast<half>(), result.ReinterpretCast<half>(),
             SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(destination[offset], result, dmaAligned);
      // Conservative ownership fences first; overlap is a later, separately
      // measured experiment once exact conversion has passed on hardware.
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, words_, parity_, floats_, constant_, rounded_, output_, canonical_, nanMask_,
      exponentMask_, gatherMask_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_query_bf16_vector_v1(GM_ADDR input, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  const int64_t count = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (count <= 0) return;
  QueryConvert converter;
  converter.Run(input, output, count);
}
