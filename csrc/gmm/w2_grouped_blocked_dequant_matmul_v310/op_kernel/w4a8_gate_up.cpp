// SPDX-License-Identifier: Apache-2.0
#include "compat_310p.h"
#include "w4a8_tile.h"
#include "w2_grouped_blocked_dequant_matmul_v310_tiling_data.h"

// Decode-only ABI: x is ONE row shared by all compact route IDs. Independent
// output buffers preserve the ordinary projection rounding and SwiGLU path.
extern "C" __global__ __aicore__ void w2_grouped_blocked_dequant_matmul_v310(
    GM_ADDR x, GM_ADDR codes, GM_ADDR ratios, GM_ADDR groupEnds, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling, GM_ADDR activationScales, GM_ADDR weightScales, GM_ADDR decodeOffsets,
    GM_ADDR upCodes, GM_ADDR upRatios, GM_ADDR upWeightScales, GM_ADDR upY) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ W2GroupedBlockedDequantMatmulKernelTilingData*>(tiling);
  if (td->nzPacked != 3 || td->numRows < 1 || td->numRows > NsW2::INT8_FUSED_MAX_ROWS) {
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "paired W4A8 requires compact decode routes"); });
    return;
  }
  AscendC::GlobalTensor<int32_t> routeIds;
  routeIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupEnds));
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  NsW2::W4A8Tile op;
  const int64_t n = td->nDim;
  const int64_t k = td->kDim;
  const int64_t scaleStride = (n / 32) * (k / 32);
  bool activationReady = false;
  for (int64_t route = 0; route < td->numRows; ++route) {
    const int64_t expert = routeIds.GetValue(route);
    // Uncomputed peer rows must be masked by the caller, as in compact v15.
    if (expert < 0 || expert >= td->numExperts) continue;
    for (int projection = 0; projection < 2; ++projection) {
      GM_ADDR bank = projection == 0 ? codes : upCodes;
      GM_ADDR blockRatios = projection == 0 ? ratios : upRatios;
      GM_ADDR outputScales = projection == 0 ? weightScales : upWeightScales;
      GM_ADDR output = projection == 0 ? y : upY;
      op.fusedActivation_ = true;
      op.InitGeometry(x, bank + expert * n * k / 2,
                      blockRatios + expert * scaleStride * sizeof(float),
                      output + route * n * sizeof(half), user, 1, n, k, 2, true, true);
      op.xInt8Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
      op.activationScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activationScales));
      op.weightScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputScales) + expert * n / 32);
      op.offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(decodeOffsets));
      op.ProcessInt8(activationReady, activationReady);
      activationReady = true;
      // Finish each projection before rebinding its weight/output pointers.
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}
