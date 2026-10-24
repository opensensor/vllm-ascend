
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QSA_SPARSE_ATTENTION_V310_H_
#define ACLNN_QSA_SPARSE_ATTENTION_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQsaSparseAttentionV310GetWorkspaceSize
 * parameters :
 * query : required
 * keyCache : required
 * valueCache : required
 * groupIndices : required
 * groupCounts : required
 * tailStarts : required
 * tailCounts : required
 * blockTable : required
 * queryStartLoc : required
 * scaleQ24 : required
 * compressRatio : optional
 * logicalKvHeads : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaSparseAttentionV310GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *keyCache,
    const aclTensor *valueCache,
    const aclTensor *groupIndices,
    const aclTensor *groupCounts,
    const aclTensor *tailStarts,
    const aclTensor *tailCounts,
    const aclTensor *blockTable,
    const aclTensor *queryStartLoc,
    int64_t scaleQ24,
    int64_t compressRatio,
    int64_t logicalKvHeads,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQsaSparseAttentionV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaSparseAttentionV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
