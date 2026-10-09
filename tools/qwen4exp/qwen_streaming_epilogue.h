// SPDX-License-Identifier: Apache-2.0
// Bounded output columns only. Qualified builtin FP16 SwiGLU remains external.
#ifndef QWEN_STREAMING_EPILOGUE_H
#define QWEN_STREAMING_EPILOGUE_H
#include "qwen_streaming_contract.h"
#if defined(__NPU_ARCH__)
  #define QWEN_EPILOGUE_DEVICE __aicore__
#else
  #define QWEN_EPILOGUE_DEVICE
#endif

namespace qwen_streaming {
constexpr uint32_t EPILOGUE_PROJECTED_COLUMNS = 1280;
constexpr uint32_t EPILOGUE_HIDDEN_COLUMNS = EPILOGUE_PROJECTED_COLUMNS / 2;
constexpr uint32_t EPILOGUE_MAX_WINDOW_TILES = BLOCKS;

struct ColumnWindowContract {
  uint32_t firstTile;
  uint32_t tileCount;
  QWEN_EPILOGUE_DEVICE constexpr bool Valid(uint32_t totalColumns) const {
    return totalColumns >= N && totalColumns <= MAX_K && totalColumns % N == 0 && tileCount > 0 &&
           tileCount <= EPILOGUE_MAX_WINDOW_TILES && firstTile < totalColumns / N &&
           tileCount <= totalColumns / N - firstTile;
  }
  QWEN_EPILOGUE_DEVICE constexpr uint32_t FirstColumn() const { return firstTile * N; }
  QWEN_EPILOGUE_DEVICE constexpr uint32_t Columns() const { return tileCount * N; }
  QWEN_EPILOGUE_DEVICE constexpr uint64_t RoutedBytes(uint32_t physicalRows) const {
    return static_cast<uint64_t>(physicalRows) * Columns() * sizeof(uint16_t);
  }
  QWEN_EPILOGUE_DEVICE constexpr uint64_t OutputElement(uint32_t physicalRow, uint32_t column) const {
    return static_cast<uint64_t>(physicalRow) * Columns() + column;
  }
  QWEN_EPILOGUE_DEVICE constexpr uint32_t GateColumn(uint32_t hiddenColumn) const { return hiddenColumn; }
  QWEN_EPILOGUE_DEVICE constexpr uint32_t UpColumn(uint32_t hiddenColumn) const {
    return EPILOGUE_HIDDEN_COLUMNS + hiddenColumn;
  }
};
// Gate and up owners may differ. The launch/stream boundary must complete the
// entire projected row before builtin activation and compact hidden packing.
// This contract does not implement an unqualified on-core nonlinear fusion.
static_assert(EPILOGUE_HIDDEN_COLUMNS % GROUP == 0, "hidden G128 quantizer boundary");
static_assert(EPILOGUE_MAX_WINDOW_TILES * N == 1024, "bounded routed-output window");
}  // namespace qwen_streaming
#undef QWEN_EPILOGUE_DEVICE
#endif
