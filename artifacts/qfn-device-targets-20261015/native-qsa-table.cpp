// SPDX-License-Identifier: Apache-2.0
#include "qsa_cube_sparse_attention_v310.h"
extern "C" __global__ __aicore__ void qwen_qsa_table_v1(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR groups, GM_ADDR counts,
                                                        GM_ADDR tail_starts, GM_ADDR tail_counts, GM_ADDR table,
                                                        GM_ADDR bounds, GM_ADDR output, GM_ADDR tiling) {
  QsaSparseAttentionV310TilingData td;
  auto* destination = reinterpret_cast<int64_t*>(&td);
  auto* source = reinterpret_cast<__gm__ int64_t*>(tiling);
  for (int i = 0; i < sizeof(td) / sizeof(int64_t); ++i) destination[i] = source[i];
  AscendC::TPipe pipe;
  NsQsaCubeSparseAttention::QsaCubeSparseAttentionV310 kernel;
  kernel.Init(q, k, v, groups, counts, tail_starts, tail_counts, table, bounds, output, &td, &pipe);
  kernel.Process();
}
