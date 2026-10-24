
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CAUSAL_CONV1D_V310_H_
#define ACLNN_CAUSAL_CONV1D_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCausalConv1dV310GetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * biasOptional : optional
 * convStates : required
 * queryStartLocOptional : optional
 * cacheIndicesOptional : optional
 * initialStateModeOptional : optional
 * numAcceptedTokensOptional : optional
 * activationMode : optional
 * padSlotId : optional
 * runMode : optional
 * stateStride : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCausalConv1dV310GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    const aclTensor *convStates,
    const aclTensor *queryStartLocOptional,
    const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional,
    const aclTensor *numAcceptedTokensOptional,
    int64_t activationMode,
    int64_t padSlotId,
    int64_t runMode,
    int64_t stateStride,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCausalConv1dV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCausalConv1dV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
