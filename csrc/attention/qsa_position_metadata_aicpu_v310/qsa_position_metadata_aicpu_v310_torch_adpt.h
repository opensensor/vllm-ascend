// SPDX-License-Identifier: Apache-2.0
#ifndef QSA_POSITION_METADATA_AICPU_V310_TORCH_ADPT_H
#define QSA_POSITION_METADATA_AICPU_V310_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor npu_qsa_position_metadata_aicpu_310(const at::Tensor& positions,
                                                 int64_t compress_ratio,
                                                 int64_t capacity,
                                                 int64_t selected_width) {
    TORCH_CHECK(positions.dim() == 1 && positions.scalar_type() == at::kInt,
                "positions must be a 1D int32 tensor");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(compress_ratio > 0 && capacity >= 0 && selected_width >= 0 &&
                    selected_width <= capacity,
                "invalid QSA position metadata geometry");
    auto metadata = at::empty({4, positions.size(0)}, positions.options());
    EXEC_NPU_CMD(aclnnQsaPositionMetadataAicpuV310, positions, metadata,
                 compress_ratio, capacity, selected_width);
    return metadata;
}
}  // namespace vllm_ascend

#endif  // QSA_POSITION_METADATA_AICPU_V310_TORCH_ADPT_H
