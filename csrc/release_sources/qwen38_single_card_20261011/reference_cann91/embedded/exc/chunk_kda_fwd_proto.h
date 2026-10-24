#ifndef CHUNK_KDA_FWD_PROTO_H_
#define CHUNK_KDA_FWD_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ChunkKdaFwd)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .INPUT(g, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .OPTIONAL_INPUT(a_log, ge::TensorType::ALL())
    .OPTIONAL_INPUT(dt_bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(initial_state, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens, ge::TensorType::ALL())
    .OPTIONAL_INPUT(chunk_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(stage_dependency, ge::TensorType::ALL())
    .OPTIONAL_INPUT(gk_fp16, ge::TensorType::ALL())
    .OPTIONAL_INPUT(beta_fp16, ge::TensorType::ALL())
    .OUTPUT(attn_out, ge::TensorType::ALL())
    .OUTPUT(final_state, ge::TensorType::ALL())
    .OUTPUT(gk, ge::TensorType::ALL())
    .OUTPUT(Aqk, ge::TensorType::ALL())
    .OUTPUT(Akk, ge::TensorType::ALL())
    .OUTPUT(w, ge::TensorType::ALL())
    .OUTPUT(u, ge::TensorType::ALL())
    .OUTPUT(qg, ge::TensorType::ALL())
    .OUTPUT(kg, ge::TensorType::ALL())
    .OUTPUT(v_new, ge::TensorType::ALL())
    .OUTPUT(h, ge::TensorType::ALL())
    .OUTPUT(qg_scaled, ge::TensorType::ALL())
    .OUTPUT(u_seed, ge::TensorType::ALL())
    .OUTPUT(score_scratch, ge::TensorType::ALL())
    .OUTPUT(score_matrices, ge::TensorType::ALL())
    .OUTPUT(stage_token, ge::TensorType::ALL())
    .ATTR(layout, String, "BSND")
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(chunk_size, Int)
    .REQUIRED_ATTR(safe_gate, Bool)
    .ATTR(lower_bound, Float, -5)
    .REQUIRED_ATTR(use_gate_in_kernel, Bool)
    .ATTR(state_v_first, Bool, false)
    .ATTR(stage, Int, -1)
    .OP_END_FACTORY_REG(ChunkKdaFwd);

}

#endif
