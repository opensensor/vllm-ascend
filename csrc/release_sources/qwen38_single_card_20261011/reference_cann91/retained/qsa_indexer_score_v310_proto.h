#ifndef QSA_INDEXER_SCORE_V310_PROTO_H_
#define QSA_INDEXER_SCORE_V310_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(QsaIndexerScoreV310)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(compressedKeyCache, ge::TensorType::ALL())
    .INPUT(blockTable, ge::TensorType::ALL())
    .INPUT(queryStartLoc, ge::TensorType::ALL())
    .INPUT(positions, ge::TensorType::ALL())
    .OUTPUT(scores, ge::TensorType::ALL())
    .ATTR(compressRatio, Int, 4)
    .OP_END_FACTORY_REG(QsaIndexerScoreV310);

}

#endif
