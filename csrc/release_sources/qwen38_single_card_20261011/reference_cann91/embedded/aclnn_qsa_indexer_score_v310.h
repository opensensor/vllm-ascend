
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QSA_INDEXER_SCORE_V310_H_
#define ACLNN_QSA_INDEXER_SCORE_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQsaIndexerScoreV310GetWorkspaceSize
 * parameters :
 * query : required
 * compressedKeyCache : required
 * blockTable : required
 * queryStartLoc : required
 * positions : required
 * compressRatio : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaIndexerScoreV310GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *compressedKeyCache,
    const aclTensor *blockTable,
    const aclTensor *queryStartLoc,
    const aclTensor *positions,
    int64_t compressRatio,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQsaIndexerScoreV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaIndexerScoreV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
