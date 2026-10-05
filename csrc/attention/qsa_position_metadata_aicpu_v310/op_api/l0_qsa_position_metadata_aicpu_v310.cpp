// SPDX-License-Identifier: Apache-2.0
#include "l0_qsa_position_metadata_aicpu_v310.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QsaPositionMetadataAicpuV310);

const aclTensor* QsaPositionMetadataAicpuV310(
    const aclTensor* positions, const aclTensor* metadata, int64_t compress_ratio,
    int64_t capacity, int64_t selected_width, aclOpExecutor* executor) {
    L0_DFX(QsaPositionMetadataAicpuV310, positions, metadata, compress_ratio,
           capacity, selected_width);
    static internal::AicpuTaskSpace space("QsaPositionMetadataAicpuV310");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        QsaPositionMetadataAicpuV310,
        OP_ATTR_NAMES({"compress_ratio", "capacity", "selected_width"}),
        OP_INPUT(positions, metadata),
        OP_ATTR(compress_ratio, capacity, selected_width));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QsaPositionMetadataAicpuV310 launch failed"),
             return nullptr);
    return metadata;
}
}  // namespace l0op
