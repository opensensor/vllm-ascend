
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QSA_GATHER_VALUE_NZ_V310_H_
#define ACLNN_QSA_GATHER_VALUE_NZ_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQsaGatherValueNzV310GetWorkspaceSize
 * parameters :
 * valueCache : required
 * groupIndices : required
 * groupCounts : required
 * tailStarts : required
 * tailCounts : required
 * blockTable : required
 * numKvHeads : required
 * headDim : required
 * transposeOutput : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaGatherValueNzV310GetWorkspaceSize(
    const aclTensor *valueCache,
    const aclTensor *groupIndices,
    const aclTensor *groupCounts,
    const aclTensor *tailStarts,
    const aclTensor *tailCounts,
    const aclTensor *blockTable,
    int64_t numKvHeads,
    int64_t headDim,
    bool transposeOutput,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQsaGatherValueNzV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQsaGatherValueNzV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
