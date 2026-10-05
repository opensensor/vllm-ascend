// SPDX-License-Identifier: Apache-2.0
#ifndef QSA_POSITION_METADATA_AICPU_V310_PROTO_H
#define QSA_POSITION_METADATA_AICPU_V310_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
REG_OP(QsaPositionMetadataAicpuV310)
    .INPUT(positions, TensorType({DT_INT32}))
    .INPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(compress_ratio, Int)
    .REQUIRED_ATTR(capacity, Int)
    .REQUIRED_ATTR(selected_width, Int)
    .OP_END_FACTORY_REG(QsaPositionMetadataAicpuV310)
}  // namespace ge

#endif  // QSA_POSITION_METADATA_AICPU_V310_PROTO_H
