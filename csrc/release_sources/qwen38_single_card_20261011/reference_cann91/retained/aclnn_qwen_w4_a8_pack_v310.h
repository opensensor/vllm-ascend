
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN_W4A8PACK_V310_H_
#define ACLNN_QWEN_W4A8PACK_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwenW4A8PackV310GetWorkspaceSize
 * parameters :
 * x : required
 * lowOut : required
 * highOut : required
 * scaleOut : required
 * sumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4A8PackV310GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *lowOut,
    const aclTensor *highOut,
    const aclTensor *scaleOut,
    const aclTensor *sumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwenW4A8PackV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4A8PackV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
