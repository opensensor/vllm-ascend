// SPDX-License-Identifier: Apache-2.0
// Packed-code handoff only: quantizer, scales and nonlinear arithmetic are unchanged.
#pragma once
#include "kernel_operator.h"
#include "glm_streaming_contract.h"

namespace GlmStreamingEpilogue {
using namespace AscendC;
constexpr uint32_t RECORD_BYTES = GlmStreamingContract::GROUP;
constexpr uint32_t GROUPS = GlmStreamingContract::COLUMNS / GlmStreamingContract::GROUP;
constexpr uint32_t QUANT_ROWS = 4;
constexpr uint32_t PAIRS = GROUPS / 2;
constexpr uint32_t PAIR_BYTES = GlmStreamingContract::OPERAND_A_SLOT_BYTES;
constexpr uint32_t TILE_BYTES = GlmStreamingContract::EPILOGUE_TILE_BYTES;
static_assert(PAIRS == 2 && PAIR_BYTES == 2048 && TILE_BYTES == 4096, "external paired-group ABI is fixed");
static_assert(2 * GlmStreamingContract::MAX_ROWS * RECORD_BYTES <= PAIR_BYTES, "valid records must fit each pair");

// Caller drains projection/nonlinear readers before lending converted scratch.
// This clear defines the entire physical tile, including all partial-tail bytes.
__aicore__ inline void Clear(LocalTensor<int8_t> tile) {
  Duplicate(tile.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), TILE_BYTES / sizeof(int16_t));
  PipeBarrier<PIPE_V>();
}

// Quantize() has completed before entry. Its four-row source is reused by the
// next batch, while tile must survive every batch until the final Store().
__aicore__ inline void Stage(LocalTensor<int8_t> tile, LocalTensor<int8_t> quantized, uint32_t row, uint32_t count) {
  if (row >= count) return;
  const uint32_t remaining = count - row;
  const uint16_t rows = static_cast<uint16_t>(remaining < QUANT_ROWS ? remaining : QUANT_ROWS);
  // One instruction gathers a group's valid 32B records from the four-row
  // quantizer output into contiguous logical rows. Strides are 32B gaps.
  const DataCopyParams copy{rows, 1, GROUPS - 1, 0};
#pragma unroll
  for (uint32_t group = 0; group < GROUPS; ++group) {
    const uint32_t source = group * RECORD_BYTES;
    const uint32_t destination = (group / 2) * PAIR_BYTES + (group % 2 * count + row) * RECORD_BYTES;
    DataCopy(tile[destination], quantized[source], copy);
  }
  // UB-to-UB copies finish before the next quantizer overwrites their source.
  // There is no per-record PIPE_ALL fence and no GM store inside this loop.
  PipeBarrier<PIPE_V>();
}

// Caller computes offsets with the existing batch/column ownership formula.
// Both offsets are distinct aligned pair regions in the unchanged compact GM
// handoff. Fence once around the two copies before lending tile to another use.
__aicore__ inline void Store(GlobalTensor<int8_t> dst, LocalTensor<int8_t> tile, uint64_t firstPairOffset,
                             uint64_t secondPairOffset) {
  PipeBarrier<PIPE_ALL>();
  DataCopy(dst[firstPairOffset], tile, PAIR_BYTES);
  DataCopy(dst[secondPairOffset], tile[PAIR_BYTES], PAIR_BYTES);
  PipeBarrier<PIPE_ALL>();
}
}  // namespace GlmStreamingEpilogue
