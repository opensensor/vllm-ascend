// SPDX-License-Identifier: Apache-2.0
// Weight reconstruction storage also has live product/activation/scale uses.
#pragma once
#include <cstdint>
namespace GlmFusedScratch {
template <uint32_t Rows, uint32_t WideCubeK, bool CompactW4>
struct Layout {
  static_assert(Rows == 16 || Rows == 32, "scratch supports the existing Cube row schedules");
  static_assert(WideCubeK == 64 || WideCubeK == 128 || WideCubeK == 256, "invalid wide Cube geometry");
  static constexpr uint32_t N = 128, NZ_K = 256, GROUP = 32, MAX_K_GROUPS = 4096 / GROUP;
  static constexpr uint32_t ELEMENTS = Rows * N;
  static constexpr uint32_t A_BYTES = Rows * 64 / 2;
  static constexpr uint32_t WIDE_A_BYTES = Rows * WideCubeK / 2;
  static constexpr uint32_t ACTIVATION_BYTES = WIDE_A_BYTES > 2 * A_BYTES ? WIDE_A_BYTES : 2 * A_BYTES;
  static constexpr uint32_t WEIGHT_VECTOR_BYTES = (NZ_K / GROUP) * N * sizeof(float);
  static constexpr uint32_t FACTOR_CACHE_BYTES = 4 * (N / GROUP) * MAX_K_GROUPS * sizeof(float);
  static constexpr uint32_t RAW_BYTES = (CompactW4 ? ACTIVATION_BYTES : N * NZ_K / 2) + 64;
  // M32 readback keeps two INT32 and two FP32 planes here. M16 uses separate
  // product buffers; its only remaining decoded_ use is eight weight vectors.
  static constexpr uint32_t DECODED_BYTES = CompactW4 && Rows == 16 ? WEIGHT_VECTOR_BYTES : N * NZ_K * sizeof(uint16_t);
  static constexpr uint32_t SCALE_PRODUCTS_OFFSET = CompactW4 ? 0 : N * NZ_K * 3 / 4;
  static constexpr uint32_t GATHERED_BYTES = SCALE_PRODUCTS_OFFSET + FACTOR_CACHE_BYTES;
  static_assert(DECODED_BYTES >= WEIGHT_VECTOR_BYTES, "weight broadcasts exceed decoded scratch");
  static_assert(Rows != 32 || DECODED_BYTES >= 4 * ELEMENTS * sizeof(float), "wide readback exceeds decoded scratch");
  static_assert(GATHERED_BYTES >= SCALE_PRODUCTS_OFFSET + WEIGHT_VECTOR_BYTES,
                "scale broadcasts exceed gathered scratch");
  static_assert(RAW_BYTES >= ACTIVATION_BYTES, "wide activation packing exceeds raw scratch");
};
}  // namespace GlmFusedScratch
