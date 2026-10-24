#ifndef QSA_GATHER_VALUE_NZ_V310_PROTO_H_
#define QSA_GATHER_VALUE_NZ_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QsaGatherValueNzV310)
    .INPUT(valueCache, ge::TensorType::ALL())
    .INPUT(groupIndices, ge::TensorType::ALL())
    .INPUT(groupCounts, ge::TensorType::ALL())
    .INPUT(tailStarts, ge::TensorType::ALL())
    .INPUT(tailCounts, ge::TensorType::ALL())
    .INPUT(blockTable, ge::TensorType::ALL())
    .OUTPUT(valueNz, ge::TensorType::ALL())
    .REQUIRED_ATTR(numKvHeads, Int)
    .REQUIRED_ATTR(headDim, Int)
    .REQUIRED_ATTR(transposeOutput, Bool)
    .OP_END_FACTORY_REG(QsaGatherValueNzV310);

}

#endif
