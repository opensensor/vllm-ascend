
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GDN_GATING_V310_H_
#define ACLNN_GDN_GATING_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGdnGatingV310GetWorkspaceSize
 * parameters :
 * a : required
 * b : required
 * negExpALogTiled : required
 * dtBiasTiled : required
 * beta : optional
 * gOut : required
 * betaOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnGatingV310GetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *negExpALogTiled,
    const aclTensor *dtBiasTiled,
    double beta,
    const aclTensor *gOut,
    const aclTensor *betaOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGdnGatingV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGdnGatingV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
