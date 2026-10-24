#ifndef QWEN_W8_ROUTE_COMBINE_V310_PROTO_H_
#define QWEN_W8_ROUTE_COMBINE_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QwenW8RouteCombineV310)
    .INPUT(sortedRoutes, ge::TensorType::ALL())
    .INPUT(inverseOrder, ge::TensorType::ALL())
    .INPUT(routeWeights, ge::TensorType::ALL())
    .INPUT(localRouteMask, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(QwenW8RouteCombineV310);

}

#endif
