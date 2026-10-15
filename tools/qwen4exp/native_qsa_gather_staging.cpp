// SPDX-License-Identifier: Apache-2.0
// Isolated exact-layout gather; the frozen headers differ only in staging width.
#if QSA_STAGING_BLOCKS == 4
  #include "qsa_gather_staging_4.h"
  #define QSA_STAGING_ENTRY qwen_qsa_gather_staging_4_v1
#elif QSA_STAGING_BLOCKS == 8
  #include "qsa_gather_staging_8.h"
  #define QSA_STAGING_ENTRY qwen_qsa_gather_staging_8_v1
#else
  #error "select exactly four or eight staged token blocks"
#endif

namespace {
constexpr int64_t MAX_QUERY_TOKENS = 64, MAX_KV_HEADS = 2, HEAD_DIM_BLOCKS = 16;
constexpr int64_t CACHE_BLOCK_SIZE = 64, SELECTED_GROUPS = 512, OUTPUT_TOKEN_BLOCKS = 129;
constexpr int64_t MAX_TABLE_ENTRIES = 4096, MAX_BLOCKS = 8;
}  // namespace

extern "C" __global__ __aicore__ void QSA_STAGING_ENTRY(GM_ADDR cache, GM_ADDR indices, GM_ADDR counts,
                                                        GM_ADDR tailStarts, GM_ADDR tailCounts, GM_ADDR table,
                                                        GM_ADDR output, GM_ADDR descriptor) {
  // dav-m200 ACLRTC requires the combined AI Core classification even for
  // vector/DMA-only work; preserve the qualified gather body unchanged.
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  AscendC::InitSocState();
  const auto raw = reinterpret_cast<__gm__ int64_t*>(descriptor);
  // Exact 11-field pinned descriptor ABI; copy into scalar local storage.
  const QsaGatherValueNzV310TilingData td{raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
                                          raw[6], raw[7], raw[8], raw[9], raw[10]};
  const int64_t blocks = AscendC::GetBlockNum();
  if (td.numTokens <= 0 || td.numTokens > MAX_QUERY_TOKENS || td.numKvHeads <= 0 || td.numKvHeads > MAX_KV_HEADS ||
      td.headDimBlocks != HEAD_DIM_BLOCKS || td.cacheHeadDimBlocks != td.numKvHeads * HEAD_DIM_BLOCKS ||
      td.cacheBlockSize != CACHE_BLOCK_SIZE || td.selectedGroupsWidth != SELECTED_GROUPS ||
      td.outputTokenBlocks != OUTPUT_TOKEN_BLOCKS || td.maxBlocksPerSequence <= 0 ||
      td.maxBlocksPerSequence > MAX_TABLE_ENTRIES || td.taskCount != td.numTokens * td.numKvHeads || blocks <= 0 ||
      blocks > MAX_BLOCKS || blocks > td.taskCount || td.tasksPerCore != (td.taskCount + blocks - 1) / blocks ||
      (td.transposeOutput != 0 && td.transposeOutput != 1))
    return;
  AscendC::TPipe pipe;
  NsQsaGatherValueNz::QsaGatherValueNzV310 op;
  op.Init(cache, indices, counts, tailStarts, tailCounts, table, output, &td, &pipe);
  op.Process();
}
