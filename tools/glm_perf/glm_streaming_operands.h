// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef GLM_STREAMING_DEVICE_API
  #define GLM_STREAMING_DEVICE_API
#endif
#include "glm_streaming_contract.h"

namespace GlmStreamingOperands {
using namespace AscendC;
constexpr uint32_t ACTIVATION_BLOCK_BYTES = GlmStreamingContract::OPERAND_A_SLOT_BYTES / 2;
constexpr uint32_t FRACTAL_ROWS = 16;
constexpr uint32_t ACTIVATION_REPEATS = GlmStreamingContract::ROWS / FRACTAL_ROWS;
constexpr uint32_t WEIGHT_REPEATS = GlmStreamingContract::COLUMNS / FRACTAL_ROWS;

// T6 owns slot state and only passes used=true after ReleaseOperands. The
// backward flag protects L0 operands through the last Cube read, including
// the final partially populated paired tile. Event IDs are private by kind.
__aicore__ inline void WaitReusable(uint32_t slot, bool used) {
  if (used) {
    const auto event = static_cast<event_t>(GlmStreamingContract::EVENT_FIRST + slot);
    WaitFlag<HardEvent::M_MTE1>(event);
    // The MTE1 -> MTE2 backward fence is separate: Cube release does not
    // itself prevent a staging copy from overwriting an old L1 reader.
    WaitFlag<HardEvent::MTE1_MTE2>(event);
  }
}

// dst already selects the cache pair or bounded staging slot. Gate's L1 cache
// is retained for up: skip this helper, and pass copiedActivation=false to Load.
// The caller acquires reused staging slots with WaitReusable before copying.
__aicore__ inline void StageActivation(LocalTensor<int8_t> dst, GlobalTensor<int8_t> src, uint64_t offset,
                                       uint32_t slot) {
  DataCopy(dst, src[offset], GlmStreamingContract::OPERAND_A_SLOT_BYTES);
  SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(GlmStreamingContract::EVENT_FIRST + slot));
}

// Preserve the existing M32 INT4 interpretation: two LoadData repeats per
// physical M32 block, not one contiguous reinterpretation of M64. Weight
// subviews select resident prepared W4 or reconstructed W2/W3 codes identically.
// Internal borrowed-argument entry lets T6 avoid constructing an OperandViews
// aggregate. The public frozen Load adapter remains available below.
__aicore__ inline void LoadDirect(const LocalTensor<int8_t>& l1a, const LocalTensor<int8_t>& l0a,
                                  const LocalTensor<int8_t>& l1b, const LocalTensor<int8_t>& l0b, uint32_t physicalRows,
                                  uint32_t slot, bool copiedActivation) {
  const auto event = static_cast<event_t>(GlmStreamingContract::EVENT_FIRST + slot);
  if (copiedActivation) WaitFlag<HardEvent::MTE2_MTE1>(event);
  LoadData2DParams load;
  load.repeatTimes = ACTIVATION_REPEATS;
  load.srcStride = 1;
  load.ifTranspose = false;
  const uint32_t blocks = physicalRows / GlmStreamingContract::ROWS;
  for (uint32_t block = 0; block < blocks; ++block)
    LoadData(l0a[block * ACTIVATION_BLOCK_BYTES].ReinterpretCast<int4b_t>(),
             l1a[block * ACTIVATION_BLOCK_BYTES].ReinterpretCast<int4b_t>(), load);
  load.repeatTimes = WEIGHT_REPEATS;
  LoadData(l0b.ReinterpretCast<int4b_t>(), l1b.ReinterpretCast<int4b_t>(), load);
  SetFlag<HardEvent::MTE1_MTE2>(event);
  SetFlag<HardEvent::MTE1_M>(event);
}

__aicore__ inline void Load(const GlmStreamingContract::OperandViews& views, uint32_t physicalRows, uint32_t slot,
                            bool copiedActivation) {
  LoadDirect(views.l1a, views.l0a, views.l1b, views.l0b, physicalRows, slot, copiedActivation);
}

// Called immediately after Mmad is issued, not after vector consumption. This
// flag carries only Cube -> operand-reuse ordering; product reuse is separate.
__aicore__ inline void ReleaseOperands(uint32_t slot) {
  SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(GlmStreamingContract::EVENT_FIRST + slot));
}

__aicore__ inline void Drain(bool slot0, bool slot1) {
  WaitReusable(0, slot0);
  WaitReusable(1, slot1);
}
}  // namespace GlmStreamingOperands
