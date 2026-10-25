// SPDX-License-Identifier: Apache-2.0
#include "compat_310p.h"
#include "w4a8_tile.h"
#include "w2_grouped_blocked_dequant_matmul_v310_tiling_data.h"

extern "C" __global__ __aicore__ void w2_grouped_blocked_dequant_matmul_v310(
    GM_ADDR x, GM_ADDR codes, GM_ADDR ratios, GM_ADDR groupEnds, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling, GM_ADDR activationScales, GM_ADDR weightScales, GM_ADDR decodeOffsets) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ W2GroupedBlockedDequantMatmulKernelTilingData*>(tiling);
  AscendC::GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupEnds));
  AscendC::GlobalTensor<int32_t> routeIds;
  routeIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupEnds));
  const bool compactRoutes = td->nzPacked == 3;
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  NsW2::W4A8Tile op;
  const int64_t n = td->nDim;
  const int64_t k = td->kDim;
  const int64_t scaleStride = (n / 32) * (k / 32);
  int64_t start = 0;
  bool tablesReady = false;
  const int64_t groups = compactRoutes ? td->numRows : td->numExperts;
  for (int64_t group = 0; group < groups; ++group) {
    const int64_t expert = compactRoutes ? routeIds.GetValue(group) : group;
    if (compactRoutes) {
      start = group;
      // Peer routes have no local projection and are masked by the caller.
      if (expert < 0 || expert >= td->numExperts) continue;
    }
    const int64_t end = compactRoutes ? group + 1 : ends.GetValue(expert);
    if (end < start || end > td->numRows) {
      ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported W4A8 grouped boundaries"); });
      return;
    }
    for (int64_t first = start; first < end; first += NsW2::W2_TILE_M) {
      const int64_t count = NsW2::MinU<int64_t>(end - first, NsW2::W2_TILE_M);
      op.fusedActivation_ = td->nzPacked >= 2;
      op.InitGeometry(x + first * k * (op.fusedActivation_ ? sizeof(half) : 1), codes + expert * n * k / 2,
                      ratios + expert * scaleStride * sizeof(float),
                      y + first * n * sizeof(half), user, count, n, k, 2, true, true);
      op.xInt8Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x) + first * k);
      op.activationScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activationScales) + first);
      op.weightScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightScales) + expert * n / 32);
      op.offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(decodeOffsets));
      op.debugDecode_ = td->codesPerByte == 0;
      op.debugGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(user));
      op.ProcessInt8(tablesReady);
      tablesReady = true;
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    start = end;
  }
}
