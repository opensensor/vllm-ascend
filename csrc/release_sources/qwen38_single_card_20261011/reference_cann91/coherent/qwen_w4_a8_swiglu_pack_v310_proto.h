#ifndef QWEN_W4_A8_SWIGLU_PACK_V310_PROTO_H_
#define QWEN_W4_A8_SWIGLU_PACK_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QwenW4A8SwigluPackV310)
    .INPUT(gate_up, ge::TensorType::ALL())
    .OUTPUT(low, ge::TensorType::ALL())
    .OUTPUT(high, ge::TensorType::ALL())
    .OUTPUT(scale, ge::TensorType::ALL())
    .OUTPUT(sum, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(QwenW4A8SwigluPackV310);

}

#endif
