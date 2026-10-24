#ifndef QSA_INDEX_CACHE_UPDATE_V310_PROTO_H_
#define QSA_INDEX_CACHE_UPDATE_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QsaIndexCacheUpdateV310)
    .INPUT(compressedKeyCache, ge::TensorType::ALL())
    .INPUT(indexKeys, ge::TensorType::ALL())
    .INPUT(queryStartLoc, ge::TensorType::ALL())
    .INPUT(slotMapping, ge::TensorType::ALL())
    .INPUT(keyNormWeight, ge::TensorType::ALL())
    .INPUT(ropeCos, ge::TensorType::ALL())
    .INPUT(ropeSin, ge::TensorType::ALL())
    .OUTPUT(compressedKeyCache, ge::TensorType::ALL())
    .ATTR(blockSize, Int, 128)
    .ATTR(compressRatio, Int, 4)
    .ATTR(rotaryDim, Int, 64)
    .ATTR(normEps, Float, 1e-06)
    .OP_END_FACTORY_REG(QsaIndexCacheUpdateV310);

}

#endif
