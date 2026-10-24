
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_QWEN_W4A8SWIGLU_PACK_V310_H_
#define ACLNN_QWEN_W4A8SWIGLU_PACK_V310_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQwenW4A8SwigluPackV310GetWorkspaceSize
 * parameters :
 * gateUp : required
 * lowOut : required
 * highOut : required
 * scaleOut : required
 * sumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4A8SwigluPackV310GetWorkspaceSize(
    const aclTensor *gateUp,
    const aclTensor *lowOut,
    const aclTensor *highOut,
    const aclTensor *scaleOut,
    const aclTensor *sumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnQwenW4A8SwigluPackV310
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQwenW4A8SwigluPackV310(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
