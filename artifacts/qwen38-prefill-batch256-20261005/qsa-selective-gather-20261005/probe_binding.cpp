// SPDX-License-Identifier: Apache-2.0
// Register an additional operator for isolated experiments; baseline binding stays loaded.

#include <torch/library.h>
#include "op_api_common.h"
#include "csrc/attention/qsa_gather_value_nz_zero_v310/qsa_gather_value_nz_zero_310_torch_adpt.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def(
      "qsa_gather_value_nz_zero_310(Tensor value_cache, Tensor group_indices, Tensor group_counts, "
      "Tensor tail_starts, Tensor tail_counts, Tensor block_table, Tensor(a!) value_nz, "
      "int num_kv_heads, int head_dim, bool transpose_output=False) -> ()");
  ops.impl("qsa_gather_value_nz_zero_310", torch::kPrivateUse1, &vllm_ascend::qsa_gather_value_nz_zero_310);
}
