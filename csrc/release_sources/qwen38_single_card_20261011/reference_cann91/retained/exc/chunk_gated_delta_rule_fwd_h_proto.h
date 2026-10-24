#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_PROTO_H_
#define CHUNK_GATED_DELTA_RULE_FWD_H_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ChunkGatedDeltaRuleFwdH)
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .INPUT(u, ge::TensorType::ALL())
    .OPTIONAL_INPUT(g, ge::TensorType::ALL())
    .OPTIONAL_INPUT(gk, ge::TensorType::ALL())
    .OPTIONAL_INPUT(inital_state, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens, ge::TensorType::ALL())
    .OPTIONAL_INPUT(chunk_indices, ge::TensorType::ALL())
    .OUTPUT(h, ge::TensorType::ALL())
    .OUTPUT(v_new, ge::TensorType::ALL())
    .OUTPUT(final_state, ge::TensorType::ALL())
    .REQUIRED_ATTR(output_final_state, Bool)
    .REQUIRED_ATTR(chunk_size, Int)
    .OP_END_FACTORY_REG(ChunkGatedDeltaRuleFwdH);

}

#endif
