
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN_W8ROUTE_COMBINE_V310_H_
#define ACLNN_QWEN_W8ROUTE_COMBINE_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwenW8RouteCombineV310GetWorkspaceSize
 * parameters :
 * sortedRoutes : required
 * inverseOrder : required
 * routeWeights : required
 * localRouteMask : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW8RouteCombineV310GetWorkspaceSize(
    const aclTensor *sortedRoutes,
    const aclTensor *inverseOrder,
    const aclTensor *routeWeights,
    const aclTensor *localRouteMask,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwenW8RouteCombineV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW8RouteCombineV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
