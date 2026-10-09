// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "glm_fused_compact_scales.h"
namespace GlmFusedDownScales {
using namespace AscendC;
constexpr uint32_t ROWS = GlmFusedCompactScales::ROWS;
constexpr uint32_t GROUPS = GlmFusedCompactScales::GROUPS_PER_TILE;
constexpr uint32_t QUANT_ROWS = 4, MIN_ROWS = 5;
constexpr uint32_t TILE_ELEMENTS = ROWS * GROUPS;
constexpr uint32_t TRANSPOSE_OFFSET = TILE_ELEMENTS;
constexpr uint32_t INDEX_OFFSET = 2 * TILE_ELEMENTS;
constexpr uint32_t STORAGE_BYTES = (INDEX_OFFSET + ROWS) * sizeof(float);
static_assert(sizeof(uint32_t) == sizeof(float), "scale indices and floats require equal storage widths");
__aicore__ inline bool GroupMajor(uint32_t count) { return count >= MIN_ROWS; }
__aicore__ inline void Prepare(LocalTensor<float> storage) {
  // Gate/up input scales are dead; quantization uses independent scratch.
  Duplicate(storage, 0.0f, TILE_ELEMENTS);
  PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void Stage(LocalTensor<float> storage, LocalTensor<float> scales, uint32_t row) {
  // All sixteen FP32 scalars, including the quantizer's partial-tail padding.
  DataCopy(storage[row * GROUPS], scales, QUANT_ROWS * GROUPS);
  PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void Finish(LocalTensor<float> storage, uint32_t count) {
  auto indices = storage.ReinterpretCast<uint32_t>()[INDEX_OFFSET];
  for (uint32_t row = 0; row < ROWS; ++row) indices.SetValue(row, (row < count ? row : 0) * GROUPS * sizeof(float));
  PipeBarrier<PIPE_ALL>();
  for (uint32_t group = 0; group < GROUPS; ++group)
    Gather(storage[TRANSPOSE_OFFSET + group * ROWS], storage[group], indices, static_cast<uint32_t>(0), ROWS);
  PipeBarrier<PIPE_ALL>();
}
}  // namespace GlmFusedDownScales
