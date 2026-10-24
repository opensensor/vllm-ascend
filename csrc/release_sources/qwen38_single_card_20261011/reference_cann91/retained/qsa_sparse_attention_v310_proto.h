#ifndef QSA_SPARSE_ATTENTION_V310_PROTO_H_
#define QSA_SPARSE_ATTENTION_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QsaSparseAttentionV310)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(keyCache, ge::TensorType::ALL())
    .INPUT(valueCache, ge::TensorType::ALL())
    .INPUT(groupIndices, ge::TensorType::ALL())
    .INPUT(groupCounts, ge::TensorType::ALL())
    .INPUT(tailStarts, ge::TensorType::ALL())
    .INPUT(tailCounts, ge::TensorType::ALL())
    .INPUT(blockTable, ge::TensorType::ALL())
    .INPUT(queryStartLoc, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .REQUIRED_ATTR(scaleQ24, Int)
    .ATTR(compressRatio, Int, 4)
    .OP_END_FACTORY_REG(QsaSparseAttentionV310);

}

#endif
