#ifndef RECURRENT_GATED_DELTA_RULE_V310_PROTO_H_
#define RECURRENT_GATED_DELTA_RULE_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(RecurrentGatedDeltaRuleV310)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .INPUT(state, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths, ge::TensorType::ALL())
    .INPUT(ssm_state_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(g, ge::TensorType::ALL())
    .OPTIONAL_INPUT(gk, ge::TensorType::ALL())
    .OPTIONAL_INPUT(num_accepted_tokens, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OUTPUT(state, ge::TensorType::ALL())
    .ATTR(scale_value, Float, 1)
    .OP_END_FACTORY_REG(RecurrentGatedDeltaRuleV310);

}

#endif
