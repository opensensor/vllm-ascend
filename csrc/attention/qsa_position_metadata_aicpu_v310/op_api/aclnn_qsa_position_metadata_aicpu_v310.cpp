// SPDX-License-Identifier: Apache-2.0
#include "aclnn_qsa_position_metadata_aicpu_v310.h"
#include "l0_qsa_position_metadata_aicpu_v310.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

extern "C" {
aclnnStatus aclnnQsaPositionMetadataAicpuV310GetWorkspaceSize(
    const aclTensor* positions, const aclTensor* metadata, int64_t compress_ratio,
    int64_t capacity, int64_t selected_width, uint64_t* workspaceSize,
    aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnQsaPositionMetadataAicpuV310,
                   DFX_IN(positions, compress_ratio, capacity, selected_width),
                   DFX_OUT(metadata));
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(positions != nullptr && metadata != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(compress_ratio > 0 && capacity >= 0 && selected_width >= 0 &&
                  selected_width <= capacity,
              ACLNN_ERR_PARAM_INVALID);
    auto result = l0op::QsaPositionMetadataAicpuV310(
        positions, metadata, compress_ratio, capacity, selected_width,
        unique_executor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = 0;
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQsaPositionMetadataAicpuV310(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQsaPositionMetadataAicpuV310);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}  // extern "C"
