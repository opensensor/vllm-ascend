// SPDX-License-Identifier: Apache-2.0
// Isolated direct entry for the repository's corrected 310P output kernel.
#include "kernel_operator.h"
struct ChunkFwdOVllmTilingData {
  int64_t shapeBatch, seqlen, kNumHead, vNumHead, kHeadDim, vHeadDim;
  int64_t chunkSize, isVariedLen, tokenBatch, dataType, gDataType;
  int64_t vWorkspaceOffset, hWorkspaceOffset, attnWorkspaceOffset;
  int64_t aftermaskWorkspaceOffset, maskWorkspaceOffset;
  float scale;
};
#include "arch20/compat_310p.h"
#include "arch20/gemm/kernel/gdn_fwd_o_kernel.hpp"
extern "C" __global__ __aicore__ void qwen_gdn_output_aligned_v1(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                                                 GM_ADDR cu, GM_ADDR chunks, GM_ADDR out,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {
  Catlass::Gemm::Kernel::GDNFwdOKernel<half, float, float> kernel;
  kernel.Init(q, k, v, h, g, cu, chunks, out, tiling, workspace);
  kernel.Process();
}
