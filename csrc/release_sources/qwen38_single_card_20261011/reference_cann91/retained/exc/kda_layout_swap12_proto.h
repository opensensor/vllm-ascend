#ifndef KDA_LAYOUT_SWAP12_PROTO_H_
#define KDA_LAYOUT_SWAP12_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(KdaLayoutSwap12)
    .INPUT(x, ge::TensorType::ALL())
    .OPTIONAL_INPUT(dependency, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(KdaLayoutSwap12);

}

#endif
