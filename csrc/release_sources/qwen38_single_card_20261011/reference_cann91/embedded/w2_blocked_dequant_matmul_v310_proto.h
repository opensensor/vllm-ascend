#ifndef W2_BLOCKED_DEQUANT_MATMUL_V310_PROTO_H_
#define W2_BLOCKED_DEQUANT_MATMUL_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(W2BlockedDequantMatmulV310)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(codes, ge::TensorType::ALL())
    .INPUT(blockScale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(W2BlockedDequantMatmulV310);

}

#endif
