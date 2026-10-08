// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "expert_core_groups.h"
#include "w2_blocked_dequant_matmul_v310.h"
#include "w2_grouped_blocked_dequant_matmul_v310_tiling_data.h"

extern "C" __global__ __aicore__ void w2_grouped_blocked_dequant_matmul_v310(
    GM_ADDR x, GM_ADDR codes, GM_ADDR blockScale, GM_ADDR groupEnds, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ W2GroupedBlockedDequantMatmulKernelTilingData*>(tiling);
  AscendC::GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupEnds));
  const int64_t n = td->nDim;
  const int64_t k = td->kDim;
  const int64_t packedK = td->codesPerByte == 3 ? k * NsW2::W3_BYTES_PER_GROUP / NsW2::W3_CODES_PER_GROUP
                                                 : k / td->codesPerByte;
  const int64_t scaleStride = (n / NsW2::W2_BLOCK_SIZE) * (k / NsW2::W2_BLOCK_SIZE);
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  NsW2::W2BlockedDequantMatmulV310Cube op;
  int64_t start = 0;
  // The tiler fixes N, K, code width, and layout for every expert in this
  // invocation. Keep the prepared UB decode tables across active experts,
  // including when an empty expert lies between them.
  bool decodeTablesReady = false;
#ifdef GLM_W2_GROUPED_EXPERT_LANES
  const auto team = NsW2::MakeExpertCoreGroup(
      AscendC::GetBlockIdx(), AscendC::GetBlockNum(), GLM_W2_GROUPED_EXPERT_LANES, td->numRows);
  unsigned activeExpert = 0;
#endif
  for (int64_t expert = 0; expert < td->numExperts; ++expert) {
    const int64_t end = ends.GetValue(expert);
    if (end < start || end > td->numRows) {
      ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid grouped W2/W3/W4 boundaries"); });
      return;
    }
    if (end > start) {
#ifdef GLM_W2_GROUPED_EXPERT_LANES
      // Assign by active ordinal, so empty experts do not leave holes in
      // team ownership. Every core observes the same device-side boundaries.
      const bool ownsExpert = activeExpert++ % team.groups == team.group;
      if (!ownsExpert) {
        start = end;
        continue;
      }
#endif
      // The shared Process reuses each decoded weight tile across its internal M tiles.
#ifdef GLM_W2_GROUPED_L1_WIDE
      bool residentShape = k % NsW2::W2_L1_WIDE_K == 0;
#ifdef GLM_W2_GROUPED_L1_ROW_REUSE
      // Beyond one two-tile window, repeated dequantization lost to the GM
      // path's once-per-expert expansion in the concentrated-route benchmark.
      residentShape = residentShape && end - start <= NsW2Rows::WINDOW_ROWS;
#else
      residentShape = residentShape && end - start <= NsW2::W2_TILE_M;
#endif
#ifdef GLM_W2_GROUPED_L1_LARGE_GROUPS
      // A narrow N tile holds the full K dimension in L1, so all M tiles of
      // a large expert group reuse a weight tile decoded just once, without
      // staging it through GM. The wide path keeps small groups.
      residentShape = residentShape || (end - start > NsW2::W2_TILE_M && k <= NsW2::W2_L1_MAX_K);
#endif
#ifdef GLM_W2_GROUPED_L1_W3_PREFILL_ONLY
      residentShape = residentShape && NsW2::AllowResidentW3Prefill(td->codesPerByte, td->numRows);
#endif
      const bool useL1 = td->nzPacked != 0 && NsW2::SupportsWideL1Codes(td->codesPerByte) && residentShape;
      op.InitGeometry(x + start * k * sizeof(half),
                      codes + expert * n * packedK,
                      blockScale + expert * scaleStride * sizeof(float),
                      y + start * n * sizeof(half), user, end - start, n, k,
                      td->codesPerByte, td->nzPacked != 0, useL1);
#elif defined(GLM_W2_GROUPED_CANDIDATE)
      // The single-route NZ path sends each decoded 32xK tile to L1 instead
      // of staging a 128xK tile through GM. Multi-route groups keep the
      // known-good GM/Cube path; the earlier blanket L1 trial was slower.
      const bool singletonNz = end - start == 1 && td->nzPacked != 0 && k <= NsW2::W2_L1_MAX_K;
      op.InitGeometry(x + start * k * sizeof(half),
                      codes + expert * n * packedK,
                      blockScale + expert * scaleStride * sizeof(float),
                      y + start * n * sizeof(half), user, end - start, n, k,
                      td->codesPerByte, td->nzPacked != 0, singletonNz);
#else
      op.InitGeometry(x + start * k * sizeof(half),
                      codes + expert * n * packedK,
                      blockScale + expert * scaleStride * sizeof(float),
                      y + start * n * sizeof(half), user, end - start, n, k,
                      td->codesPerByte, td->nzPacked != 0);
#endif
#ifdef GLM_W2_GROUPED_EXPERT_LANES
      op.SetTileOwnership(team.lane, team.lanes);
#endif
      op.Process(decodeTablesReady);
      decodeTablesReady = true;
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    start = end;
  }
}
