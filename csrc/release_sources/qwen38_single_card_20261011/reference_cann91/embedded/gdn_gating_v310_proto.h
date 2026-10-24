#ifndef GDN_GATING_V310_PROTO_H_
#define GDN_GATING_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GdnGatingV310)
    .INPUT(a, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .INPUT(negExpALogTiled, ge::TensorType::ALL())
    .INPUT(dtBiasTiled, ge::TensorType::ALL())
    .OUTPUT(g, ge::TensorType::ALL())
    .OUTPUT(betaOut, ge::TensorType::ALL())
    .ATTR(beta, Float, 1)
    .OP_END_FACTORY_REG(GdnGatingV310);

}

#endif
