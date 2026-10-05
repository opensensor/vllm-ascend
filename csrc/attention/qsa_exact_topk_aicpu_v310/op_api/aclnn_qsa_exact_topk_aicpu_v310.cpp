// SPDX-License-Identifier: Apache-2.0
#include "aclnn_qsa_exact_topk_aicpu_v310.h"
#include "l0_qsa_exact_topk_aicpu_v310.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

extern "C" {
aclnnStatus aclnnQsaExactTopkAicpuV310GetWorkspaceSize(
    const aclTensor* scores, const aclTensor* indices, int64_t topk,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnQsaExactTopkAicpuV310,
                   DFX_IN(scores, topk), DFX_OUT(indices));
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(scores != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(topk > 0, ACLNN_ERR_PARAM_INVALID);
    auto result = l0op::QsaExactTopkAicpuV310(scores, indices, topk,
                                               unique_executor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = 0;
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQsaExactTopkAicpuV310(void* workspace, uint64_t workspaceSize,
                                         aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQsaExactTopkAicpuV310);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}  // extern "C"
