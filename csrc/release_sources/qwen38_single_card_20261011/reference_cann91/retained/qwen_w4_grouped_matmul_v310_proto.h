#ifndef QWEN_W4_GROUPED_MATMUL_V310_PROTO_H_
#define QWEN_W4_GROUPED_MATMUL_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QwenW4GroupedMatmulV310)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(codes, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .INPUT(offset, ge::TensorType::ALL())
    .INPUT(group_ends, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(QwenW4GroupedMatmulV310);

}

#endif
