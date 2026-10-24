#ifndef QWEN_W4_A8_INT4_DOWN_REDUCE_V310_PROTO_H_
#define QWEN_W4_A8_INT4_DOWN_REDUCE_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QwenW4A8Int4DownReduceV310)
    .INPUT(low, ge::TensorType::ALL())
    .INPUT(high, ge::TensorType::ALL())
    .INPUT(activation_scale, ge::TensorType::ALL())
    .INPUT(activation_sum, ge::TensorType::ALL())
    .INPUT(codes, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .INPUT(offset, ge::TensorType::ALL())
    .INPUT(weight_sum, ge::TensorType::ALL())
    .INPUT(route_ids, ge::TensorType::ALL())
    .INPUT(route_weights, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(QwenW4A8Int4DownReduceV310);

}

#endif
