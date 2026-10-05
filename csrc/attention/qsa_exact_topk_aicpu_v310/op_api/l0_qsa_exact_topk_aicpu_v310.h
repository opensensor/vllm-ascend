// SPDX-License-Identifier: Apache-2.0
#ifndef L0_QSA_EXACT_TOPK_AICPU_V310_H
#define L0_QSA_EXACT_TOPK_AICPU_V310_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* QsaExactTopkAicpuV310(const aclTensor* scores,
                                        const aclTensor* indices,
                                        int64_t topk, aclOpExecutor* executor);
}  // namespace l0op

#endif  // L0_QSA_EXACT_TOPK_AICPU_V310_H
