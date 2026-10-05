// SPDX-License-Identifier: Apache-2.0
#ifndef QSA_EXACT_TOPK_AICPU_V310_PROTO_H
#define QSA_EXACT_TOPK_AICPU_V310_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
REG_OP(QsaExactTopkAicpuV310)
    .INPUT(scores, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .REQUIRED_ATTR(top_k, Int)
    .OP_END_FACTORY_REG(QsaExactTopkAicpuV310)
}  // namespace ge

#endif  // QSA_EXACT_TOPK_AICPU_V310_PROTO_H
