// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "glm_streaming_contract.h"
#include "glm_streaming_scale_gather.h"

namespace GlmStreamingProducts {
using namespace AscendC;
using namespace GlmStreamingContract;
constexpr uint32_t FRACTAL_COLUMNS = 16;
constexpr uint32_t VECTOR_FLOATS = 64;
constexpr uint32_t BLOCK_BYTES = 32;

// V_M releases CO1 after readback, not the UB raw slot. The issue-loop owner
// must finish that slot's Consume before another ReadBack writes it.
__aicore__ inline void WaitReusable(uint32_t slot, bool used) {
  if (used) WaitFlag<HardEvent::V_M>(slot == 0 ? EVENT_ID0 : EVENT_ID1);
}
__aicore__ inline void ReadBack(LocalTensor<int32_t> raw, LocalTensor<int32_t> co1, uint32_t physicalRows,
                                uint32_t slot) {
  WaitFlag<HardEvent::M_V>(slot == 0 ? EVENT_ID0 : EVENT_ID1);
  const DataCopyParams copy{COLUMNS / FRACTAL_COLUMNS, static_cast<uint16_t>(physicalRows / FRACTAL_COLUMNS), 0, 0};
  DataCopyEnhancedParams enhanced;
  enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
  DataCopy(raw, co1, copy, enhanced);
  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_M>(slot == 0 ? EVENT_ID0 : EVENT_ID1);
}

// Internal direct-argument entry avoids constructing/returning a ProductViews
// aggregate in the caller. Public Consume remains the frozen T1 adapter.
template <bool GatherScales = false>
__aicore__ inline void ConsumeDirect(const LocalTensor<int32_t>& raw, const LocalTensor<float>& converted,
                                     const LocalTensor<float>& factors, const LocalTensor<float>& rowBroadcast,
                                     const LocalTensor<float>& accumulator, const LocalTensor<float>& sx0,
                                     const LocalTensor<float>& sx1, const LocalTensor<float>& weight0,
                                     const LocalTensor<float>& weight1, const LocalTensor<uint32_t>& rowIndices,
                                     uint32_t count, uint32_t physicalRows, bool groupMajorScales) {
  // Convert valid products directly from N16-fractal raw storage into the
  // row-major plane. Avoid generic UB-to-UB DataCopy dispatch and never touch
  // padded rows. Independent raw slots remain read-only through consumption.
  const UnaryRepeatParams extract{1, 1, COLUMNS / 8, FRACTAL_COLUMNS / 8};
  const BinaryRepeatParams broadcast{1, 1, 0, COLUMNS / 8, 0, 1};
  // Preserve independent K32 group order even when the next Cube pair is in
  // flight. The second product starts at logical row count, never fixed M32.
  for (uint32_t part = 0; part < 2; ++part) {
    // All eight casts use the same mask. Subsequent broadcast/arithmetic APIs
    // set their own masks, so establish it again for each independent group.
    SetMaskNorm();
    SetVectorMask<float>(FRACTAL_COLUMNS);
    for (uint32_t strip = 0; strip < COLUMNS / FRACTAL_COLUMNS; ++strip)
      Cast<float, int32_t, false>(converted[strip * FRACTAL_COLUMNS],
                                  raw[(strip * physicalRows + part * count) * FRACTAL_COLUMNS], RoundMode::CAST_NONE,
                                  static_cast<uint64_t>(FRACTAL_COLUMNS), count, extract);
    PipeBarrier<PIPE_V>();
    const auto& sx = part == 0 ? sx0 : sx1;
    const auto& weight = part == 0 ? weight0 : weight1;
    if constexpr (GatherScales) {
      // Prepared once per batch; one bit-preserving Gather replaces the
      // dav-2002 FP32 Brcb expansion on every independent K32 product.
      GlmStreamingScaleGather::Broadcast(rowBroadcast, sx, rowIndices);
    } else {
      if (groupMajorScales) {
        Brcb(rowBroadcast, sx, ROWS / 8, {1, 8});
      } else {
        Gather(factors, sx, rowIndices, static_cast<uint32_t>(0), ROWS);
        PipeBarrier<PIPE_V>();
        Brcb(rowBroadcast, factors, ROWS / 8, {1, 8});
      }
      PipeBarrier<PIPE_V>();
    }
    for (uint32_t column = 0; column < COLUMNS; column += VECTOR_FLOATS)
      Mul(factors[column], weight[column], rowBroadcast, static_cast<uint64_t>(VECTOR_FLOATS), count, broadcast);
    PipeBarrier<PIPE_V>();
    // These remain separate FP32 operations. FMA or reassociation changes
    // the qualified scale/product rounding and cancellation behavior.
    Mul(converted, converted, factors, count * COLUMNS);
    PipeBarrier<PIPE_V>();
    Add(accumulator, accumulator, converted, count * COLUMNS);
    PipeBarrier<PIPE_V>();
  }
}
__aicore__ inline void Consume(const ProductViews& views, uint32_t count, uint32_t physicalRows,
                               bool groupMajorScales) {
  ConsumeDirect(views.raw, views.converted, views.factors, views.rowBroadcast, views.accumulator, views.sx0, views.sx1,
                views.weight0, views.weight1, views.rowIndices, count, physicalRows, groupMajorScales);
}
__aicore__ inline void Drain(bool slot0, bool slot1) {
  WaitReusable(0, slot0);
  WaitReusable(1, slot1);
}
}  // namespace GlmStreamingProducts
