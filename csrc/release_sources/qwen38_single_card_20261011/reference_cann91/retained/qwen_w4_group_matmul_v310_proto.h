#ifndef QWEN_W4_GROUP_MATMUL_V310_PROTO_H_
#define QWEN_W4_GROUP_MATMUL_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QwenW4GroupMatmulV310)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(codes, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .INPUT(offset, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(tiled, Bool, false)
    .OP_END_FACTORY_REG(QwenW4GroupMatmulV310);

}

#endif
