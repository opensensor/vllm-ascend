#ifndef QWEN4_EXP_PLE_DECODE_V310_PROTO_H_
#define QWEN4_EXP_PLE_DECODE_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Qwen4ExpPleDecodeV310)
    .INPUT(projected, ge::TensorType::ALL())
    .INPUT(hidden, ge::TensorType::ALL())
    .INPUT(normKeyWeight, ge::TensorType::ALL())
    .INPUT(normQueryWeight, ge::TensorType::ALL())
    .INPUT(normConvWeight, ge::TensorType::ALL())
    .INPUT(currentConvWeight, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .ATTR(normEps, Float, 1e-06)
    .OP_END_FACTORY_REG(Qwen4ExpPleDecodeV310);

}

#endif
