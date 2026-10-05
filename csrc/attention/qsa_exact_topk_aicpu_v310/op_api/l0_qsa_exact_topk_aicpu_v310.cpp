// SPDX-License-Identifier: Apache-2.0
#include "l0_qsa_exact_topk_aicpu_v310.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QsaExactTopkAicpuV310);

const aclTensor* QsaExactTopkAicpuV310(const aclTensor* scores,
                                        const aclTensor* indices,
                                        int64_t topk, aclOpExecutor* executor) {
    L0_DFX(QsaExactTopkAicpuV310, scores, indices, topk);
    static internal::AicpuTaskSpace space("QsaExactTopkAicpuV310");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        QsaExactTopkAicpuV310, OP_ATTR_NAMES({"top_k"}),
        OP_INPUT(scores, indices), OP_ATTR(topk));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QsaExactTopkAicpuV310 launch failed"),
             return nullptr);
    return indices;
}
}  // namespace l0op
