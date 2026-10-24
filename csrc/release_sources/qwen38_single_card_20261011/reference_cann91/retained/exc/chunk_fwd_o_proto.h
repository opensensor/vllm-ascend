#ifndef CHUNK_FWD_O_PROTO_H_
#define CHUNK_FWD_O_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ChunkFwdO)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .INPUT(h, ge::TensorType::ALL())
    .INPUT(g, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens, ge::TensorType::ALL())
    .OPTIONAL_INPUT(chunk_offsets, ge::TensorType::ALL())
    .OUTPUT(o, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(chunk_size, Int)
    .OP_END_FACTORY_REG(ChunkFwdO);

}

#endif
