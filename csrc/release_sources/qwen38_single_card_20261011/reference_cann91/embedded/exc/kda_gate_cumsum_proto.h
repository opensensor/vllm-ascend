#ifndef KDA_GATE_CUMSUM_PROTO_H_
#define KDA_GATE_CUMSUM_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(KdaGateCumsum)
    .INPUT(g, ge::TensorType::ALL())
    .OPTIONAL_INPUT(a_log, ge::TensorType::ALL())
    .OPTIONAL_INPUT(dt_bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens, ge::TensorType::ALL())
    .OUTPUT(gk, ge::TensorType::ALL())
    .REQUIRED_ATTR(chunk_size, Int)
    .REQUIRED_ATTR(use_gate_in_kernel, Bool)
    .REQUIRED_ATTR(safe_gate, Bool)
    .REQUIRED_ATTR(lower_bound, Float)
    .ATTR(layout, String, "BSND")
    .OP_END_FACTORY_REG(KdaGateCumsum);

}

#endif
