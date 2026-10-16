// SPDX-License-Identifier: Apache-2.0
// Offline compiler-builtin experiment, pinned to CANN 9.1 / dav-2002.
#pragma once
#include "kernel_operator.h"
#include "glm_streaming_contract.h"
#include "glm_streaming_scale_gather.h"
#if !defined(GLM_STREAMING_POINTER_CPU_MODEL) && (!defined(__NPU_ARCH__) || __NPU_ARCH__ != 2002)
  #error "streaming pointer products require the pinned dav-2002 compiler or the dedicated CPU model"
#endif

namespace GlmStreamingProductsPointer {
using namespace AscendC;
using namespace GlmStreamingContract;
constexpr uint32_t FRACTAL_COLUMNS = 16, VECTOR_FLOATS = 64, FLOATS_PER_BLOCK = 8;
constexpr uint8_t UNIT_STRIDE = 1, ZERO_STRIDE = 0;
constexpr uint8_t ROW_BLOCKS = COLUMNS / FLOATS_PER_BLOCK;
constexpr uint8_t FRACTAL_BLOCKS = FRACTAL_COLUMNS / FLOATS_PER_BLOCK;
constexpr uint8_t VECTOR_BLOCKS = VECTOR_FLOATS / FLOATS_PER_BLOCK;
static_assert(ROWS == 32 && COLUMNS == 128 && MAX_ROWS == 31, "pointer consumer requires frozen M32/N128 rows");
static_assert(COLUMNS / VECTOR_FLOATS * MAX_ROWS <= 255, "contiguous vector repeats exceed the byte field");

// The caller prepares expanded byte indices once per batch and owns all slot
// events. Raw products remain read-only; converted/factors/accumulator are
// independent. No allocation, extra scratch, descriptor slicing or FMA.
__aicore__ inline void ConsumeDirect(const LocalTensor<int32_t>& raw, const LocalTensor<float>& converted,
                                     const LocalTensor<float>& factors, const LocalTensor<float>& rowBroadcast,
                                     const LocalTensor<float>& accumulator, const LocalTensor<float>& sx0,
                                     const LocalTensor<float>& sx1, const LocalTensor<float>& weight0,
                                     const LocalTensor<float>& weight1, const LocalTensor<uint32_t>& rowIndices,
                                     uint32_t count, uint32_t physicalRows, bool groupMajorScales) {
  // Group layout is already encoded in rowIndices; retain the T4 signature.
  (void)groupMajorScales;
  auto* rawBase = reinterpret_cast<__ubuf__ int32_t*>(raw.GetPhyAddr());
  auto* convertedBase = reinterpret_cast<__ubuf__ float*>(converted.GetPhyAddr());
  auto* factorBase = reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr());
  auto* broadcastBase = reinterpret_cast<__ubuf__ float*>(rowBroadcast.GetPhyAddr());
  auto* accumulatorBase = reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr());
  auto* weightBase0 = reinterpret_cast<__ubuf__ float*>(weight0.GetPhyAddr());
  auto* weightBase1 = reinterpret_cast<__ubuf__ float*>(weight1.GetPhyAddr());
  const uint8_t rows = static_cast<uint8_t>(count);
  const uint8_t contiguousRepeats = static_cast<uint8_t>(count * COLUMNS / VECTOR_FLOATS);
  for (uint32_t part = 0; part < 2; ++part) {
    SetMaskNorm();
    SetVectorMask<float>(FRACTAL_COLUMNS);
    for (uint32_t strip = 0; strip < COLUMNS / FRACTAL_COLUMNS; ++strip)
      vconv_s322f32(convertedBase + strip * FRACTAL_COLUMNS,
                    rawBase + (strip * physicalRows + part * count) * FRACTAL_COLUMNS, rows, UNIT_STRIDE, UNIT_STRIDE,
                    ROW_BLOCKS, FRACTAL_BLOCKS);
    PipeBarrier<PIPE_V>();
    const auto& sx = part == 0 ? sx0 : sx1;
    GlmStreamingScaleGather::Broadcast(rowBroadcast, sx, rowIndices);
    auto* weights = part == 0 ? weightBase0 : weightBase1;
    SetMaskNorm();
    SetVectorMask<float>(VECTOR_FLOATS);
    for (uint32_t column = 0; column < COLUMNS; column += VECTOR_FLOATS)
      vmul(factorBase + column, weights + column, broadcastBase, rows, UNIT_STRIDE, UNIT_STRIDE, ZERO_STRIDE,
           ROW_BLOCKS, ZERO_STRIDE, UNIT_STRIDE);
    PipeBarrier<PIPE_V>();
    vmul(convertedBase, convertedBase, factorBase, contiguousRepeats, UNIT_STRIDE, UNIT_STRIDE, UNIT_STRIDE,
         VECTOR_BLOCKS, VECTOR_BLOCKS, VECTOR_BLOCKS);
    PipeBarrier<PIPE_V>();
    vadd(accumulatorBase, accumulatorBase, convertedBase, contiguousRepeats, UNIT_STRIDE, UNIT_STRIDE, UNIT_STRIDE,
         VECTOR_BLOCKS, VECTOR_BLOCKS, VECTOR_BLOCKS);
    PipeBarrier<PIPE_V>();
  }
  // The old final count-mode Add restored both mask words to full. Preserve
  // that caller boundary, including later half-vector users of the high word.
  ResetMask();
}
}  // namespace GlmStreamingProductsPointer
