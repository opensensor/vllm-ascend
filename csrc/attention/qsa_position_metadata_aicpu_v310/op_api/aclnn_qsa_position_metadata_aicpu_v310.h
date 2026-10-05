// SPDX-License-Identifier: Apache-2.0
#ifndef ACLNN_QSA_POSITION_METADATA_AICPU_V310_H
#define ACLNN_QSA_POSITION_METADATA_AICPU_V310_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnQsaPositionMetadataAicpuV310GetWorkspaceSize(
    const aclTensor* positions, const aclTensor* metadata, int64_t compress_ratio,
    int64_t capacity, int64_t selected_width, uint64_t* workspaceSize,
    aclOpExecutor** executor);
__attribute__((visibility("default"))) aclnnStatus aclnnQsaPositionMetadataAicpuV310(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_QSA_POSITION_METADATA_AICPU_V310_H
