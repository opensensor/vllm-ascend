// SPDX-License-Identifier: Apache-2.0
// Host-only entry control-flow test. Stubs do not emulate CANN arithmetic,
// synchronization, or L1 storage; those still require a device comparison.
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define __global__
#define __aicore__
#define __gm__
#define GM_ADDR uint8_t*
#define KERNEL_TASK_TYPE_DEFAULT(kind)
#define ASCENDC_ASSERT(condition, details) assert(condition)
using half = uint16_t;
constexpr int PIPE_ALL = 0;

struct W2GroupedBlockedDequantMatmulKernelTilingData {
  int64_t numRows, numExperts, nDim, kDim, codesPerByte, nzPacked;
};

namespace AscendC {
template <typename T>
struct GlobalTensor {
  T* data = nullptr;
  void SetGlobalBuffer(T* value) { data = value; }
  T GetValue(int64_t index) { return data[index]; }
};
inline GM_ADDR GetUserWorkspace(GM_ADDR address) { return address; }
template <int>
void PipeBarrier() {}
}  // namespace AscendC

namespace NsW2 {
constexpr uint32_t INT8_FUSED_MAX_ROWS = 4;
struct W4A8Tile {
  bool fusedActivation_ = false;
  bool prepared = false;
  GM_ADDR firstInput = nullptr;
  GM_ADDR input = nullptr;
  GM_ADDR codes = nullptr;
  GM_ADDR ratios = nullptr;
  GM_ADDR output = nullptr;
  AscendC::GlobalTensor<int8_t> xInt8Gm_;
  AscendC::GlobalTensor<float> activationScaleGm_, weightScaleGm_;
  AscendC::GlobalTensor<uint32_t> offsetsGm_;
  void InitGeometry(GM_ADDR x, GM_ADDR w, GM_ADDR s, GM_ADDR y, GM_ADDR, int64_t rows, int64_t n, int64_t k,
                    int64_t packing, bool nz, bool l1) {
    assert(rows == 1 && n == 128 && k == 1024 && packing == 2 && nz && l1);
    input = x;
    codes = w;
    ratios = s;
    output = y;
  }
  void ProcessInt8(bool reuseTables, bool reuseActivation) {
    assert(fusedActivation_ && reuseTables == prepared && reuseActivation == prepared);
    if (prepared) assert(firstInput == input);
    firstInput = input;
    std::cout << "CALL " << reinterpret_cast<uintptr_t>(input) << ' ' << reinterpret_cast<uintptr_t>(codes) << ' '
              << reinterpret_cast<uintptr_t>(ratios) << ' ' << reinterpret_cast<uintptr_t>(output) << ' '
              << reinterpret_cast<uintptr_t>(weightScaleGm_.data) << ' ' << reuseActivation << '\n';
    prepared = true;
  }
};
}  // namespace NsW2

// The Python test copies the unmodified candidate entry beside this harness.
#include "entry.cpp"

int main() {
  std::vector<uint8_t> gateCodes(2 * 128 * 1024 / 2), upCodes(gateCodes.size());
  std::vector<float> gateRatios(2 * 4 * 32), upRatios(gateRatios.size());
  std::vector<float> gateScales(8), upScales(8);
  std::vector<half> x(1024), otherX(1024), gate(4 * 128), up(gate.size());
  uint32_t offsets[1] = {0};
  auto address = [](auto* ptr) { return reinterpret_cast<GM_ADDR>(ptr); };
  for (const auto& ids : std::vector<std::vector<int32_t>>{{2, 1, 0, 1}, {0}, {2, 2, -1, 2}, {1, 1, 0, 0}}) {
    // Alternate inputs so each launch must prepare new activation state.
    auto* input = ids.size() == 1 ? otherX.data() : x.data();
    W2GroupedBlockedDequantMatmulKernelTilingData tiling{static_cast<int64_t>(ids.size()), 2, 128, 1024, 2, 3};
    std::cout << "BASE " << reinterpret_cast<uintptr_t>(input) << ' ' << reinterpret_cast<uintptr_t>(gateCodes.data())
              << ' ' << reinterpret_cast<uintptr_t>(upCodes.data()) << ' '
              << reinterpret_cast<uintptr_t>(gateRatios.data()) << ' ' << reinterpret_cast<uintptr_t>(upRatios.data())
              << ' ' << reinterpret_cast<uintptr_t>(gate.data()) << ' ' << reinterpret_cast<uintptr_t>(up.data()) << ' '
              << reinterpret_cast<uintptr_t>(gateScales.data()) << ' ' << reinterpret_cast<uintptr_t>(upScales.data())
              << '\n';
    w2_grouped_blocked_dequant_matmul_v310(address(input), gateCodes.data(), address(gateRatios.data()),
                                           address(const_cast<int32_t*>(ids.data())), address(gate.data()),
                                           address(gate.data()), address(&tiling), address(gateScales.data()),
                                           address(gateScales.data()), address(offsets), upCodes.data(),
                                           address(upRatios.data()), address(upScales.data()), address(up.data()));
  }
}
