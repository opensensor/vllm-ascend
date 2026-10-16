// SPDX-License-Identifier: Apache-2.0
// Fixed bounded-slot ABI. External N128/M32/K32 layouts remain unchanged.
#pragma once
#include <cstdint>

namespace GlmStreamingContract {
constexpr uint32_t VERSION = 2;
constexpr uint32_t ROWS = 32, MAX_ROWS = 31, COLUMNS = 128, GROUP = 32;
constexpr uint32_t PAIR_K = 64, WEIGHT_TILE_K = 256, MAX_K = 4096;
constexpr uint32_t SLOTS = 2, EVENT_FIRST = 0;
constexpr uint32_t ELEMENTS = ROWS * COLUMNS;
constexpr uint32_t OPERAND_A_SLOT_BYTES = 2 * ROWS * PAIR_K / 2;
constexpr uint32_t OPERAND_B_SLOT_BYTES = COLUMNS * PAIR_K / 2;
constexpr uint32_t PRODUCT_SLOT_BYTES = 2 * ELEMENTS * sizeof(int32_t);
constexpr uint32_t PRODUCT_STORAGE_BYTES = SLOTS * PRODUCT_SLOT_BYTES;
constexpr uint32_t CONVERTED_BYTES = ELEMENTS * sizeof(float);
constexpr uint32_t FACTORS_OFFSET = 0;
constexpr uint32_t BROADCAST_OFFSET = ELEMENTS * sizeof(float);
constexpr uint32_t RETAINED_INDEX_OFFSET = COLUMNS * WEIGHT_TILE_K * 3 / 4;
constexpr uint32_t MASK_BYTES = 32768;
constexpr uint32_t RETAINED_EXPERT_END_OFFSET = RETAINED_INDEX_OFFSET + ROWS * sizeof(uint32_t);
constexpr uint32_t RETAINED_EXPERT_END_BYTES = 288 * sizeof(int64_t);
constexpr uint32_t SCALE_GATHER_INDEX_OFFSET = RETAINED_EXPERT_END_OFFSET + RETAINED_EXPERT_END_BYTES;
constexpr uint32_t SCALE_GATHER_INDEX_BYTES = ROWS * 8 * sizeof(uint32_t);
constexpr uint32_t GROUP_SCALE_SOURCE_OFFSET = SCALE_GATHER_INDEX_OFFSET + SCALE_GATHER_INDEX_BYTES;
constexpr uint32_t GROUP_SCALE_SOURCE_BYTES = ROWS * sizeof(uint32_t);
constexpr uint32_t EPILOGUE_TILE_BYTES = 2 * OPERAND_A_SLOT_BYTES;
constexpr uint32_t SDK_UB_RESERVE = 8192;
static_assert(PRODUCT_STORAGE_BYTES == 65536, "products must fit dead decoded scratch");
static_assert(BROADCAST_OFFSET + ROWS * 8 * sizeof(float) <= RETAINED_INDEX_OFFSET,
              "vector factors must preserve retained indices");
static_assert(EPILOGUE_TILE_BYTES <= CONVERTED_BYTES, "handoff must fit dead conversion plane");
static_assert(RETAINED_EXPERT_END_OFFSET == 24704 && SCALE_GATHER_INDEX_OFFSET == 27008 &&
                  GROUP_SCALE_SOURCE_OFFSET == 28032,
              "retained routing and scale Gather intervals must match the frozen M32 layout");
static_assert(GROUP_SCALE_SOURCE_OFFSET + GROUP_SCALE_SOURCE_BYTES <= MASK_BYTES,
              "retained scale Gather metadata exceeds mask scratch");
enum class SlotState : uint8_t { Free, OperandsReady, CubeIssued, ReadBack, Consumed };
}  // namespace GlmStreamingContract

// Worker helper signatures are shared before the central Projection integration.
// Define only in device/helper-stub compilation, not CPU geometry compilation.
#ifdef GLM_STREAMING_DEVICE_API
  #include "kernel_operator.h"
namespace GlmStreamingContract {
struct OperandViews {
  AscendC::LocalTensor<int8_t> l1a, l0a, l1b, l0b;
};
struct ProductViews {
  AscendC::LocalTensor<int32_t> raw;
  AscendC::LocalTensor<float> converted, factors, rowBroadcast, accumulator;
  AscendC::LocalTensor<float> sx0, sx1, weight0, weight1;
  AscendC::LocalTensor<uint32_t> rowIndices;
};
}  // namespace GlmStreamingContract
namespace GlmStreamingOperands {
__aicore__ inline void WaitReusable(uint32_t slot, bool used);
__aicore__ inline void StageActivation(AscendC::LocalTensor<int8_t> dst, AscendC::GlobalTensor<int8_t> src,
                                       uint64_t offset, uint32_t slot);
__aicore__ inline void Load(const GlmStreamingContract::OperandViews& views, uint32_t physicalRows, uint32_t slot,
                            bool copiedActivation);
__aicore__ inline void ReleaseOperands(uint32_t slot);
__aicore__ inline void Drain(bool slot0, bool slot1);
}  // namespace GlmStreamingOperands
namespace GlmStreamingProducts {
__aicore__ inline void WaitReusable(uint32_t slot, bool used);
__aicore__ inline void ReadBack(AscendC::LocalTensor<int32_t> raw, AscendC::LocalTensor<int32_t> co1,
                                uint32_t physicalRows, uint32_t slot);
__aicore__ inline void Consume(const GlmStreamingContract::ProductViews& views, uint32_t count, uint32_t physicalRows,
                               bool groupMajorScales);
__aicore__ inline void Drain(bool slot0, bool slot1);
}  // namespace GlmStreamingProducts
namespace GlmStreamingEpilogue {
__aicore__ inline void Clear(AscendC::LocalTensor<int8_t> tile);
__aicore__ inline void Stage(AscendC::LocalTensor<int8_t> tile, AscendC::LocalTensor<int8_t> quantized, uint32_t row,
                             uint32_t count);
__aicore__ inline void Store(AscendC::GlobalTensor<int8_t> dst, AscendC::LocalTensor<int8_t> tile,
                             uint64_t firstPairOffset, uint64_t secondPairOffset);
}  // namespace GlmStreamingEpilogue
#endif
