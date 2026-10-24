
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QSA_INDEX_CACHE_UPDATE_V310_H_
#define ACLNN_QSA_INDEX_CACHE_UPDATE_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQsaIndexCacheUpdateV310GetWorkspaceSize
 * parameters :
 * compressedKeyCacheRef : required
 * indexKeys : required
 * queryStartLoc : required
 * slotMapping : required
 * keyNormWeight : required
 * ropeCos : required
 * ropeSin : required
 * blockSize : optional
 * compressRatio : optional
 * rotaryDim : optional
 * normEps : optional
 * compressedKeyCacheRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaIndexCacheUpdateV310GetWorkspaceSize(
    aclTensor *compressedKeyCacheRef,
    const aclTensor *indexKeys,
    const aclTensor *queryStartLoc,
    const aclTensor *slotMapping,
    const aclTensor *keyNormWeight,
    const aclTensor *ropeCos,
    const aclTensor *ropeSin,
    int64_t blockSize,
    int64_t compressRatio,
    int64_t rotaryDim,
    double normEps,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQsaIndexCacheUpdateV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaIndexCacheUpdateV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
