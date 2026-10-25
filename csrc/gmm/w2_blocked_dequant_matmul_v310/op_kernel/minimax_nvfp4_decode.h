// SPDX-License-Identifier: Apache-2.0
#ifndef MINIMAX_NVFP4_DECODE_H
#define MINIMAX_NVFP4_DECODE_H

#include "kernel_operator.h"

// Source codes and per-row K16 E4M3 scales stay byte-packed. A per-expert
// lookup includes global scaling and final FP16 rounding, avoiding both
// requantization and device-dependent division/rounding during unpack.
class MiniMaxNvfp4Decode {
 public:
  static constexpr uint32_t VALUES = 4096;
  static constexpr uint32_t CODE_BYTES = VALUES / 2;
  static constexpr uint32_t SCALES = VALUES / 16;
  static constexpr uint32_t LOOKUP = 4096;
  static constexpr uint32_t SCRATCH_BYTES = 81920;

  __aicore__ inline void Init(AscendC::LocalTensor<uint8_t> scratch) {
    raw_ = scratch;
    scaleRaw_ = scratch[2048];
    fields_ = scratch[4096].ReinterpretCast<half>();
    decoded_ = scratch[12288].ReinterpretCast<half>();
    indices_ = scratch[20480].ReinterpretCast<int32_t>();
    scaleIndices_ = scratch[36864].ReinterpretCast<int32_t>();
    broadcast_ = scratch[38912].ReinterpretCast<int32_t>();
    offsets_ = scratch[55296].ReinterpretCast<uint32_t>();
    lookup_ = scratch[71680].ReinterpretCast<half>();
    for (uint32_t i = 0; i < VALUES; ++i) {
      offsets_.SetValue(i, ((i / 256) * 16 + i % 16) * sizeof(int32_t));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Bind(GM_ADDR codes, GM_ADDR scales, GM_ADDR lookup) {
    codes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(codes));
    scales_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(scales));
    lookupGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(lookup));
    loaded_ = false;
  }

  __aicore__ inline void LoadLookup() {
    AscendC::DataCopy(lookup_, lookupGm_, LOOKUP);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
  }

  __aicore__ inline AscendC::LocalTensor<half> Decode(int64_t tile) {
    using namespace AscendC;
    if (!loaded_) {
      LoadLookup();
      loaded_ = true;
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopy(raw_, codes_[tile * CODE_BYTES], CODE_BYTES);
    DataCopy(scaleRaw_, scales_[tile * SCALES], SCALES);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(fields_, raw_, RoundMode::CAST_NONE, CODE_BYTES);
    PipeBarrier<PIPE_V>();
    // Biased nearest rounding is exact floor(byte / 16) for all 256 bytes.
    Muls(decoded_, fields_, static_cast<half>(1.0f / 16.0f), CODE_BYTES);
    PipeBarrier<PIPE_V>();
    Adds(decoded_, decoded_, static_cast<half>(-15.0f / 32.0f), CODE_BYTES);
    PipeBarrier<PIPE_V>();
    Cast(indices_, decoded_, RoundMode::CAST_RINT, CODE_BYTES);
    PipeBarrier<PIPE_V>();
    Cast(fields_[CODE_BYTES], indices_, RoundMode::CAST_NONE, CODE_BYTES);
    PipeBarrier<PIPE_V>();
    Muls(decoded_, fields_[CODE_BYTES], static_cast<half>(16), CODE_BYTES);
    PipeBarrier<PIPE_V>();
    Sub(fields_, fields_, decoded_, CODE_BYTES);
    PipeBarrier<PIPE_V>();
    Cast(indices_, fields_, RoundMode::CAST_RINT, VALUES);
    Cast(decoded_, scaleRaw_, RoundMode::CAST_NONE, SCALES);
    PipeBarrier<PIPE_V>();
    Cast(scaleIndices_, decoded_, RoundMode::CAST_RINT, SCALES);
    PipeBarrier<PIPE_V>();
    Muls(indices_, indices_, static_cast<int32_t>(sizeof(half)), VALUES);
    Muls(scaleIndices_, scaleIndices_, static_cast<int32_t>(16 * sizeof(half)), SCALES);
    PipeBarrier<PIPE_V>();
    Gather(broadcast_, scaleIndices_, offsets_, static_cast<uint32_t>(0), VALUES);
    PipeBarrier<PIPE_V>();
    Add(indices_, indices_, broadcast_, VALUES);
    PipeBarrier<PIPE_V>();
    Gather(decoded_, lookup_, indices_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), VALUES);
    PipeBarrier<PIPE_V>();
    return decoded_;
  }

 private:
  bool loaded_ = false;
  AscendC::GlobalTensor<uint8_t> codes_, scales_;
  AscendC::GlobalTensor<half> lookupGm_;
  AscendC::LocalTensor<uint8_t> raw_, scaleRaw_;
  AscendC::LocalTensor<half> fields_, decoded_, lookup_;
  AscendC::LocalTensor<int32_t> indices_, scaleIndices_, broadcast_;
  AscendC::LocalTensor<uint32_t> offsets_;
};
#endif
