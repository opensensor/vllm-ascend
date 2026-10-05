// SPDX-License-Identifier: Apache-2.0
#ifndef L0_QSA_POSITION_METADATA_AICPU_V310_H
#define L0_QSA_POSITION_METADATA_AICPU_V310_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* QsaPositionMetadataAicpuV310(
    const aclTensor* positions, const aclTensor* metadata, int64_t compress_ratio,
    int64_t capacity, int64_t selected_width, aclOpExecutor* executor);
}  // namespace l0op

#endif  // L0_QSA_POSITION_METADATA_AICPU_V310_H
