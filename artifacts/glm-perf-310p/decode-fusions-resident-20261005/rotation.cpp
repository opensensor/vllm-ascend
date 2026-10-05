// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t DIM = 128;
constexpr int64_t TILE = 512;
constexpr int64_t STAGES = 7;
constexpr float NORMALIZE = 0.08838834764831844f;
class Rotation {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR offsets, GM_ADDR signs,
                             GM_ADDR output, int64_t elements, bool bf16) {
    in_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    offsets_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    signs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(signs));
    pipe_.InitBuffer(storage_, TILE * (6 * sizeof(float) + 2 * sizeof(half)) + TILE);
    auto value = storage_.Get<float>();
    auto left = value[TILE];
    auto right = value[2 * TILE];
    auto sign = value[3 * TILE];
    auto idxLeft = value[4 * TILE].ReinterpretCast<uint32_t>();
    auto idxRight = value[5 * TILE].ReinterpretCast<uint32_t>();
    auto inputHalf = storage_.Get<half>()[12 * TILE];
    auto outputHalf = inputHalf[TILE];
    auto mask = storage_.Get<uint8_t>()[TILE * (6 * sizeof(float) + 2 * sizeof(half))];
    for (int64_t base = GetBlockIdx() * TILE; base < elements; base += GetBlockNum() * TILE) {
      if (bf16) {
        // Gather 16-bit words into [zero, BF16] pairs: exact FP32 widening
        // without a BF16 cast or scalar loop on this architecture.
        Duplicate(inputHalf, static_cast<half>(0), TILE);
        DataCopy(outputHalf, in_[base], TILE);
        DataCopy(idxLeft, offsets_[STAGES * 2 * TILE], 2 * TILE);
        PipeBarrier<PIPE_ALL>();
        Gather(value.ReinterpretCast<half>(), inputHalf, idxLeft, static_cast<uint32_t>(0), 2 * TILE);
      } else {
        DataCopy(inputHalf, in_[base], TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(value, inputHalf, RoundMode::CAST_NONE, TILE);
      }
      PipeBarrier<PIPE_ALL>();
      for (int64_t stage = 0; stage < STAGES; ++stage) {
        DataCopy(idxLeft, offsets_[stage * 2 * TILE], TILE);
        DataCopy(idxRight, offsets_[(stage * 2 + 1) * TILE], TILE);
        DataCopy(sign, signs_[stage * TILE], TILE);
        PipeBarrier<PIPE_ALL>();
        Gather(left, value, idxLeft, static_cast<uint32_t>(0), TILE);
        Gather(right, value, idxRight, static_cast<uint32_t>(0), TILE);
        PipeBarrier<PIPE_V>();
        Mul(right, right, sign, TILE);
        PipeBarrier<PIPE_V>();
        Add(value, left, right, TILE);
        PipeBarrier<PIPE_ALL>();
      }
      Muls(value, value, NORMALIZE, TILE);
      PipeBarrier<PIPE_ALL>();
      auto bits = value.ReinterpretCast<uint32_t>();
      for (int64_t i = 0; i < TILE; ++i) {
        uint32_t b = bits.GetValue(i);
        if ((b & 0x7f800000U) == 0x7f800000U && (b & 0x007fffffU)) {
          b = (b & 0x80000000U) | 0x7fc00000U;
        } else {
          b = (b + 0x7fffU + ((b >> 16) & 1U)) & 0xffff0000U;
        }
        bits.SetValue(i, b);
      }
      PipeBarrier<PIPE_ALL>();
      // Match the qualified 310P FP16 cast: saturate overflow/Inf, NaN -> +65504.
      Mins(value, value, 65504.0f, TILE);
      PipeBarrier<PIPE_V>();
      Maxs(value, value, -65504.0f, TILE);
      PipeBarrier<PIPE_V>();
      Compare(mask, value, value, CMPMODE::EQ, TILE);
      Duplicate(left, 65504.0f, TILE);
      PipeBarrier<PIPE_V>();
      Select(value, mask, value, left, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
      PipeBarrier<PIPE_V>();
      Cast(outputHalf, value, RoundMode::CAST_NONE, TILE);
      PipeBarrier<PIPE_ALL>();
      DataCopy(out_[base], outputHalf, TILE);
      PipeBarrier<PIPE_ALL>();
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> in_, out_;
  GlobalTensor<uint32_t> offsets_;
  GlobalTensor<float> signs_;
};
}
extern "C" __global__ __aicore__ void glm_kpool_rotation_v1(
    GM_ADDR input, GM_ADDR offsets, GM_ADDR signs, GM_ADDR output, GM_ADDR tiling) {
  AscendC::InitSocState();
  const auto data = reinterpret_cast<__gm__ int64_t*>(tiling);
  Rotation op;
  op.Run(input, offsets, signs, output, data[0], data[1] != 0);
}
