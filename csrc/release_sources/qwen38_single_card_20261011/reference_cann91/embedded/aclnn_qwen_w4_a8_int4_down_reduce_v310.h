
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN_W4A8INT4DOWN_REDUCE_V310_H_
#define ACLNN_QWEN_W4A8INT4DOWN_REDUCE_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwenW4A8Int4DownReduceV310GetWorkspaceSize
 * parameters :
 * low : required
 * high : required
 * activationScale : required
 * activationSum : required
 * codes : required
 * scale : required
 * offset : required
 * weightSum : required
 * routeIds : required
 * routeWeights : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4A8Int4DownReduceV310GetWorkspaceSize(
    const aclTensor *low,
    const aclTensor *high,
    const aclTensor *activationScale,
    const aclTensor *activationSum,
    const aclTensor *codes,
    const aclTensor *scale,
    const aclTensor *offset,
    const aclTensor *weightSum,
    const aclTensor *routeIds,
    const aclTensor *routeWeights,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwenW4A8Int4DownReduceV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4A8Int4DownReduceV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
