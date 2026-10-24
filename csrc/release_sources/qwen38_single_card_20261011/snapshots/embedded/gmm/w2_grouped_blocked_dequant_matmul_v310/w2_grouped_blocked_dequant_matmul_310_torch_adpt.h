// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef W2_GROUPED_BLOCKED_DEQUANT_MATMUL_310_TORCH_ADPT_H
#define W2_GROUPED_BLOCKED_DEQUANT_MATMUL_310_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor npu_w2_grouped_blocked_dequant_matmul_310(
    const at::Tensor& x, const at::Tensor& codes,
    const at::Tensor& block_scale, const at::Tensor& group_ends) {
  // Peer-owned routes sort after the last local group. Start from zeros so
  // those rows remain exact zero without another device-side scatter kernel.
  at::Tensor out = at::zeros({x.size(0), codes.size(1)}, x.options());
  EXEC_NPU_CMD(aclnnW2GroupedBlockedDequantMatmulV310, x, codes, block_scale,
               group_ends, out);
  return out;
}
}  // namespace vllm_ascend
#endif
