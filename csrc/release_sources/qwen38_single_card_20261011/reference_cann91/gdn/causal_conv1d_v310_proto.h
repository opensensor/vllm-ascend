#ifndef CAUSAL_CONV1D_V310_PROTO_H_
#define CAUSAL_CONV1D_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(CausalConv1dV310)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .INPUT(convStates, ge::TensorType::ALL())
    .OPTIONAL_INPUT(queryStartLoc, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cacheIndices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(initialStateMode, ge::TensorType::ALL())
    .OPTIONAL_INPUT(numAcceptedTokens, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(activationMode, Int, 0)
    .ATTR(padSlotId, Int, -1)
    .ATTR(runMode, Int, 0)
    .ATTR(stateStride, Int, 0)
    .OP_END_FACTORY_REG(CausalConv1dV310);

}

#endif
