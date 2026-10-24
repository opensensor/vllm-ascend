#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_PROTO_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AddRmsNormDynamicQuant)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(residual, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OPTIONAL_INPUT(smooth_scale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(smooth_scale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(y1, ge::TensorType::ALL())
    .OUTPUT(y2, ge::TensorType::ALL())
    .OUTPUT(x, ge::TensorType::ALL())
    .OUTPUT(scale1, ge::TensorType::ALL())
    .OUTPUT(scale2, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 1e-06)
    .ATTR(output_mask, ListBool, {})
    .ATTR(dst_type, Int, 2)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuant);

}

#endif
