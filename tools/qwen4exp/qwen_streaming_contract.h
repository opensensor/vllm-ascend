// SPDX-License-Identifier: Apache-2.0
// Generated from streaming_memory.py; byte contract only, not hardware qualification.
#ifndef QWEN_STREAMING_CONTRACT_H
#define QWEN_STREAMING_CONTRACT_H
#include <stdint.h>
#if defined(__NPU_ARCH__)
  #define QWEN_DEVICE __aicore__
#else
  #define QWEN_DEVICE
#endif
namespace qwen_streaming {
constexpr const char* CONTRACT_SHA256 = "4f36a03afebcfe5f9548ef1b1032f8dc6f2c6bff3a14edc54ba7a823906966c9";
constexpr uint32_t ABI_VERSION = 1;
constexpr uint32_t M = 16;
constexpr uint32_t N = 128;
constexpr uint32_t GROUP = 128;
constexpr uint32_t K0 = 64;
constexpr uint32_t BLOCK = 16;
constexpr uint32_t LANES = 8;
constexpr uint32_t MAX_K = 2560;
constexpr uint32_t MAX_GROUPS = 20;
constexpr uint32_t SLOTS = 2;
constexpr uint32_t BLOCKS = 8;
constexpr uint32_t CONTROL_EVENT = 2;
constexpr uint32_t ALIGNMENT = 32;
constexpr uint32_t QUANTIZER_BYTES = 32768;
constexpr uint32_t PACKED_ACTIVATION_OFFSET = 0;
constexpr uint32_t PACKED_ACTIVATION_BYTES = 4096;
constexpr uint32_t PACKED_ACTIVATION_SLOT_BYTES = 2048;
constexpr uint32_t RAW_PRODUCT_OFFSET = 4096;
constexpr uint32_t RAW_PRODUCT_BYTES = 32768;
constexpr uint32_t RAW_PRODUCT_SLOT_BYTES = 16384;
constexpr uint32_t FLOAT_PRODUCT_OFFSET = 36864;
constexpr uint32_t FLOAT_PRODUCT_BYTES = 32768;
constexpr uint32_t FLOAT_PRODUCT_SLOT_BYTES = 16384;
constexpr uint32_t ACCUMULATOR_OFFSET = 69632;
constexpr uint32_t ACCUMULATOR_BYTES = 8192;
constexpr uint32_t WEIGHT_METADATA_OFFSET = 77824;
constexpr uint32_t WEIGHT_METADATA_BYTES = 15360;
constexpr uint32_t METADATA_STAGE_OFFSET = 93184;
constexpr uint32_t METADATA_STAGE_BYTES = 3072;
constexpr uint32_t METADATA_STAGE_SLOT_BYTES = 1536;
constexpr uint32_t ACTIVATION_METADATA_OFFSET = 96256;
constexpr uint32_t ACTIVATION_METADATA_BYTES = 2048;
constexpr uint32_t ACTIVATION_METADATA_SLOT_BYTES = 1024;
constexpr uint32_t PROJECTED_OUTPUT_OFFSET = 98304;
constexpr uint32_t PROJECTED_OUTPUT_BYTES = 4096;
constexpr uint32_t PERSISTENT_GATE_OFFSET = 102400;
constexpr uint32_t PERSISTENT_GATE_BYTES = 4096;
constexpr uint32_t QUANTIZER_SCRATCH_OFFSET = 106496;
constexpr uint32_t QUANTIZER_SCRATCH_BYTES = 32768;
constexpr uint32_t ENDS_OFFSET = 139264;
constexpr uint32_t ENDS_BYTES = 256;
constexpr uint32_t ROUTE_IDS_OFFSET = 139520;
constexpr uint32_t ROUTE_IDS_BYTES = 512;
constexpr uint32_t ACTIVATION_L1_OFFSET = 0;
constexpr uint32_t ACTIVATION_L1_BYTES = 4096;
constexpr uint32_t ACTIVATION_L1_SLOT_BYTES = 2048;
constexpr uint32_t RESIDENT_WEIGHT_OFFSET = 4096;
constexpr uint32_t RESIDENT_WEIGHT_BYTES = 163840;
constexpr uint32_t ACTIVATION_L0_OFFSET = 0;
constexpr uint32_t ACTIVATION_L0_BYTES = 4096;
constexpr uint32_t ACTIVATION_L0_SLOT_BYTES = 2048;
constexpr uint32_t WEIGHT_L0_OFFSET = 0;
constexpr uint32_t WEIGHT_L0_BYTES = 16384;
constexpr uint32_t WEIGHT_L0_SLOT_BYTES = 8192;
constexpr uint32_t CUBE_PRODUCT_OFFSET = 0;
constexpr uint32_t CUBE_PRODUCT_BYTES = 16384;
constexpr uint32_t UB_USED = 140032;
constexpr uint32_t UB_LIMIT = 253952;
static_assert(UB_USED <= UB_LIMIT, "UB streaming overcommit");
constexpr uint32_t L1_USED = 167936;
constexpr uint32_t L1_LIMIT = 1048576;
static_assert(L1_USED <= L1_LIMIT, "L1 streaming overcommit");
constexpr uint32_t L0A_USED = 4096;
constexpr uint32_t L0A_LIMIT = 65536;
static_assert(L0A_USED <= L0A_LIMIT, "L0A streaming overcommit");
constexpr uint32_t L0B_USED = 16384;
constexpr uint32_t L0B_LIMIT = 65536;
static_assert(L0B_USED <= L0B_LIMIT, "L0B streaming overcommit");
constexpr uint32_t L0C_USED = 16384;
constexpr uint32_t L0C_LIMIT = 262144;
static_assert(L0C_USED <= L0C_LIMIT, "L0C streaming overcommit");
// Each direction independently uses slot ID 0/1; never share a live generation.
QWEN_DEVICE constexpr uint32_t EventId(uint32_t slot) { return slot; }
QWEN_DEVICE constexpr uint32_t Slot(uint32_t group) { return group % SLOTS; }
QWEN_DEVICE constexpr uint32_t ProductIndex(uint32_t strip, uint32_t limb, uint32_t row, uint32_t column) {
  return strip * (2 * M * BLOCK) + limb * (M * BLOCK) + row * BLOCK + column;
}
QWEN_DEVICE constexpr uint32_t AccumulatorIndex(uint32_t strip, uint32_t row, uint32_t column) {
  return strip * (M * BLOCK) + row * BLOCK + column;
}
// Kernel helper contract (member methods; no host/device launch ABI implied):
// Produce(uint32_t slot, int64_t row, uint32_t group, uint32_t live_rows)
// IssueCube(uint32_t slot): both packed limbs, m=2*M; CO1 must be free.
// ReadBack(uint32_t slot): M_V -> DataCopy -> V_M, releases single CO1.
// Consume(uint32_t slot, uint32_t live_rows): stable G128 correction/add.
// Store(int64_t row, uint32_t live_rows): FP16 projection boundary + MTE3_V.
}  // namespace qwen_streaming
#undef QWEN_DEVICE
#endif
