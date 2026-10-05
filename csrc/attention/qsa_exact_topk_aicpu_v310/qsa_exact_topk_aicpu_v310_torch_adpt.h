// SPDX-License-Identifier: Apache-2.0
#ifndef QSA_EXACT_TOPK_AICPU_V310_TORCH_ADPT_H
#define QSA_EXACT_TOPK_AICPU_V310_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor npu_qsa_exact_topk_aicpu_310(const at::Tensor& scores, int64_t topk) {
    TORCH_CHECK(scores.dim() == 2 && scores.scalar_type() == at::kFloat,
                "scores must be a 2D float32 tensor");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(topk > 0 && topk <= scores.size(1), "topk must be in [1, groups]");
    auto indices = at::empty({scores.size(0), topk}, scores.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnQsaExactTopkAicpuV310, scores, indices, topk);
    return indices;
}
}  // namespace vllm_ascend

#endif  // QSA_EXACT_TOPK_AICPU_V310_TORCH_ADPT_H
