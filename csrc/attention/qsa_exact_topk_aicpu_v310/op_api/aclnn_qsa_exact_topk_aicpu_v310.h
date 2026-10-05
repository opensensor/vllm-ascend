// SPDX-License-Identifier: Apache-2.0
#ifndef ACLNN_QSA_EXACT_TOPK_AICPU_V310_H
#define ACLNN_QSA_EXACT_TOPK_AICPU_V310_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnQsaExactTopkAicpuV310GetWorkspaceSize(
    const aclTensor* scores, const aclTensor* indices, int64_t topk,
    uint64_t* workspace_size, aclOpExecutor** executor);
__attribute__((visibility("default"))) aclnnStatus aclnnQsaExactTopkAicpuV310(
    void* workspace, uint64_t workspace_size, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_QSA_EXACT_TOPK_AICPU_V310_H
